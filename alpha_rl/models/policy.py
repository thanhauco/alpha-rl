import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from typing import Tuple

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

def layer_init(layer, std=1.414, bias_const=0.0):
    import numpy as np
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> Normal:
        mean = torch.sigmoid(self.net(x))
        # Prevent numerical explosion or degenerate zero variances
        clamped_log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = clamped_log_std.exp().expand_as(mean)
        return Normal(mean, std)

class ValueCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actor = GaussianActor(obs_dim, action_dim, hidden_dim)
        self.critic = ValueCritic(obs_dim, hidden_dim)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        dist = self.actor(x)
        if action is None:
            action = dist.sample()
            action = torch.clamp(action, 0.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(x)
        return action, log_prob, entropy, value
