import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional
from alpha_rl.models.policy import ActorCritic

class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.ac = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            dist = self.ac.actor(x)
            action = dist.mean if deterministic else dist.sample()
            action = torch.clamp(action, 0.0, 1.0)
            return action.squeeze(0).cpu().numpy()
