from collections import namedtuple
import datetime, os, shutil
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from itertools import count
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from model import A2CActorCritic
import envs

Transition = namedtuple("Transition", ["state", "action", "log_prob", "value", "reward", "done"])

class A2CTrainer:
    def __init__(self, env, model, n_episodes=5000, gamma=0.99, lr=1e-4):
        self.env = env
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.all_rewards = []
        self.all_losses = []

        folder_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        self.folder_path = os.path.join("results", folder_name)
        if os.path.exists(self.folder_path):
            shutil.rmtree(self.folder_path)
        os.makedirs(self.folder_path)

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_action(self, state):
        state = state.unsqueeze(0)
        logits, value = self.model(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1.0 - done)
            returns.insert(0, R)
        return torch.tensor(returns, device=self.device)

    def train(self):
        for ep in range(self.n_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, device=self.device)
            done = False
            episode = []

            total_reward = 0
            while not done:
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, *_ = self.env.step(envs.Action(action))
                next_state = torch.tensor(next_state, device=self.device)

                total_reward += reward
                episode.append(Transition(state, action, log_prob, value, reward, done))
                state = next_state

            with torch.no_grad():
                _, last_value = self.model(state.unsqueeze(0))
            returns = self.compute_returns([e.reward for e in episode], [e.done for e in episode], last_value)

            policy_loss = 0
            value_loss = 0
            for i, transition in enumerate(episode):
                advantage = returns[i] - transition.value
                policy_loss += -transition.log_prob * advantage.detach()
                value_loss += advantage.pow(2)

            loss = policy_loss + value_loss * 0.5
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.all_rewards.append(total_reward)
            self.all_losses.append(loss.item())

            print(f"[EP {ep}] total_reward: {total_reward}")
            if ep % 50 == 0:
                self.save_model(ep)
                self.save_gif(ep)

        final_path = os.path.join(self.folder_path, "a2c.pth")
        torch.save(self.model.state_dict(), final_path)
        print(f"Modèle final sauvegardé dans : {final_path}")

        self.plot_metrics()
        self.env.close()

    def save_model(self, ep):
        path = os.path.join(self.folder_path, f"model-{ep}.pth")
        torch.save(self.model.state_dict(), path)

    def save_gif(self, ep):
        frames = [Image.fromarray(obs, "RGB") for obs in self.env.frames]
        path = os.path.join(self.folder_path, f"episode-{ep}.gif")
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)

    def plot_metrics(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.all_rewards, label="Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward per Episode")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.all_losses, label="Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("A2C Loss")
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.folder_path, "metrics_plot.png")
        plt.savefig(plot_path)
        plt.close()

        df = pd.DataFrame({
            "episode": list(range(len(self.all_rewards))),
            "reward": self.all_rewards,
            "loss": self.all_losses
        })
        csv_path = os.path.join(self.folder_path, "metrics.csv")
        df.to_csv(csv_path, index=False)

        print("📈 Courbes et CSV enregistrés dans :", self.folder_path)


if __name__ == "__main__":
    env = gym.make("Env-v0", render_mode="rgb_array", game_mode="train")
    env = envs.Wrapper(env, k=4)

    in_channels = env.observation_space.shape[0]
    out_channels = env.action_space.n

    model = A2CActorCritic(in_channels, out_channels)
    trainer = A2CTrainer(env, model)
    trainer.train()
