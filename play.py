import gymnasium as gym
import pygame
import torch
import envs
import argparse
import matplotlib.pyplot as plt
import numpy as np
from train_ppo_optimized import PPOActorCritic

from model import DQN, A2CActorCritic  


def human_play():
    env = gym.make("Env-v0", render_mode="human")
    obs, _ = env.reset()

    total_reward = 0.0
    n_frames = 0
    while True:
        n_frames += 1
        userInput = pygame.key.get_pressed()
        action = envs.Action.STAND
        if userInput[pygame.K_UP] or userInput[pygame.K_SPACE]:
            action = envs.Action.JUMP
        elif userInput[pygame.K_DOWN]:
            action = envs.Action.DUCK

        obs, reward, terminated, _, _ = env.step(action)

        total_reward += float(reward)
        if terminated:
            break

    print(f"Total reward: {total_reward}, number of frames: {n_frames}")

    env.close()

    # Show image of the last frame
    plt.imshow(obs)
    plt.show()


def play_with_model(env: envs.Wrapper, policy_net, device: torch.device, is_ppo: bool) -> float:
    state, _ = env.reset()
    state = torch.tensor(state, device=device, dtype=torch.float32)

    total_reward = 0.0
    while True:
        with torch.no_grad():
            if is_ppo:
                logits, _ = policy_net(state.unsqueeze(0))
                action = torch.argmax(logits, dim=1)[0]
            else:
                action = policy_net(state.unsqueeze(0)).max(dim=1)[1][0]

        state, reward, terminated, _, _ = env.step(action)
        state = torch.tensor(state, device=device, dtype=torch.float32)

        total_reward += float(reward)
        if terminated:
            break

    return total_reward



def ai_play(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Tentative de chargement complet d’un objet de modèle
        policy_net = torch.load(model_path, map_location=device,  weights_only=False)
        is_ppo = isinstance(policy_net, (PPOActorCritic, A2CActorCritic))
        if not is_ppo and not isinstance(policy_net, DQN):
            raise ValueError("Modèle inconnu.")
    except:
        # Chargement d’un state_dict
        print("Chargement state_dict détecté, tentative avec PPO puis A2C...")
        env = gym.make("Env-v0", render_mode="rgb_array", game_mode="train")
        env = envs.Wrapper(env, k=4)
        in_channels = env.observation_space.shape[0]
        out_channels = env.action_space.n
        state_dict = torch.load(model_path, map_location=device)

        # ⚠️ Détection par clé de state_dict ou test manuel
        try:
            policy_net = PPOActorCritic(in_channels, out_channels).to(device)
            policy_net.load_state_dict(state_dict)
            is_ppo = True
        except:
            try:
                policy_net = A2CActorCritic(in_channels, out_channels).to(device)
                policy_net.load_state_dict(state_dict)
                is_ppo = True  # même logique d’appel
            except Exception as e:
                raise ValueError("Impossible de charger le modèle : ni PPO ni A2C.") from e
        env.close()

    policy_net.eval()

    env = gym.make("Env-v0", render_mode="human")
    env = envs.Wrapper(env)

    total_reward = play_with_model(env, policy_net, device, is_ppo)

    print(f"Total reward: {total_reward}, number of frames: {len(env.frames)}")

    env.close()
    plt.imshow(env.frames[-1])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=["human", "ai"])
    parser.add_argument("-m", "--model_path")

    args = parser.parse_args()
    if args.type == "human":
        human_play()
    else:
        ai_play(args.model_path)
