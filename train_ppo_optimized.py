import datetime
import os
import shutil
from collections import deque
from itertools import count

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import envs


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class PPOActorCritic(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

        self.apply(init_weights)

    def forward(self, x):
        x = self.shared(x)
        x = self.fc(x)
        return self.policy_head(x), self.value_head(x)


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []


class PPOTrainer:
    def __init__(
        self,
        env,
        model,
        n_episodes=1300,
        gamma=0.99,
        lr=2.5e-4,
        gae_lambda=0.95,
        clip_param=0.2,
        update_epochs=4,
        minibatch_size=64,
        rollout_steps=2048,
    ):
        self.env = env
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.rollout_steps = rollout_steps
        self.n_episodes = n_episodes

        self.buffer = RolloutBuffer()

    def select_action(self, state):
        logits, value = self.model(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy().mean(), value

    def compute_gae(self, rewards, dones, values, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self):
        states = torch.stack(self.buffer.states).detach()
        actions = torch.tensor(self.buffer.actions).to(self.device)
        logprobs = torch.tensor(self.buffer.logprobs).to(self.device)
        returns = []
        values = torch.stack(self.buffer.values).squeeze().detach()

        next_state = self.buffer.states[-1]
        with torch.no_grad():
            _, next_value = self.model(next_state.unsqueeze(0))
            next_value = next_value.item()

        advantages = self.compute_gae(
            self.buffer.rewards,
            self.buffer.dones,
            values.cpu().numpy().tolist(),
            next_value,
        )

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + values

        for _ in range(self.update_epochs):
            for i in range(0, len(states), self.minibatch_size):
                end = i + self.minibatch_size
                mb_idx = slice(i, end)

                logits, value = self.model(states[mb_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logprobs = dist.log_prob(actions[mb_idx])
                entropy = dist.entropy().mean()

                ratio = (new_logprobs - logprobs[mb_idx]).exp()
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages[mb_idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(returns[mb_idx], value.squeeze())
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.buffer.clear()

    def plot_metrics(self, rewards, losses):
        os.makedirs("results/ppo", exist_ok=True)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(rewards, label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward per Episode")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(losses, label="Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("PPO Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig("results/ppo/metrics_plot.png")
        plt.close()

        df = pd.DataFrame({
            "episode": list(range(len(rewards))),
            "reward": rewards,
            "loss": losses
        })
        df.to_csv("results/ppo/metrics.csv", index=False)
        print("📊 Courbes et CSV enregistrés dans le dossier 'results/'")

    def train(self):
        global_step = 0
        all_rewards = []
        all_losses = []

        for episode in range(self.n_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            episode_reward = 0
            self.env.frames.clear()

            for _ in range(self.rollout_steps):
                global_step += 1
                action, logprob, entropy, value = self.select_action(state.unsqueeze(0))
                next_state, reward, done, *_ = self.env.step(envs.Action(action.item()))
                next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32)

                self.buffer.states.append(state)
                self.buffer.actions.append(action.item())
                self.buffer.logprobs.append(logprob.item())
                self.buffer.rewards.append(reward)
                self.buffer.dones.append(done)
                self.buffer.values.append(value)

                state = next_state
                episode_reward += reward

                if done:
                    print(f"Episode {episode}, total reward: {episode_reward}")
                    break

            loss = self.update()
            all_rewards.append(episode_reward)
            all_losses.append(loss)

            if episode % 50 == 0:
                self.save_obs_result(episode, self.env.frames)
                self.save_model_weights()

        self.plot_metrics(all_rewards, all_losses)
        self.env.close()



    def save_obs_result(self, episode_i: int, obs_arr: list):
        frames = [Image.fromarray(obs, "RGB") for obs in obs_arr]
        file_path = os.path.join("results", f"episode-{episode_i}.gif")
        frames[0].save(
            file_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=100,
            loop=0,
        )

    def save_model_weights(self):
        file_path = os.path.join("C:\\Users\\MSI\\Desktop\\pac PPO QL\\Pacman-RL-master\\Pacman-RL-master\\chrome-dino-game-rl", "ppo.pth")
        torch.save(self.model, file_path)


if __name__ == "__main__":
    env = gym.make("Env-v0", render_mode="rgb_array", game_mode="train")
    env = envs.Wrapper(env, k=4)

    obs_space = env.observation_space.shape
    in_channels = obs_space[0]
    out_channels = env.action_space.n

    model = PPOActorCritic(in_channels, out_channels)
    trainer = PPOTrainer(env, model)
    trainer.train()
