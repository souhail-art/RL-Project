import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(3072, 256), nn.ReLU(), nn.Linear(256, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class A2CActorCritic(nn.Module):
    def __init__(self, input_channels, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Taille fixe basée sur (4, 84, 84) => sortie de convolution aplatie = 3072
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 512),
            nn.ReLU()
        )

        # Têtes pour policy (actor) et value (critic)
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0  # Normalisation si les entrées sont des images uint8
        x = self.conv(x)
        x = self.fc(x)
        return self.actor(x), self.critic(x)