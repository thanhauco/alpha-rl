import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional, Tuple
from alpha_rl.models.policy import ActorCritic

class PPOAgent:
    """Proximal Policy Optimization with Generalized Advantage Estimation (GAE)."""
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
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

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(n_steps)):
            next_val = next_value if t == n_steps - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        obs_b: torch.Tensor,
        act_b: torch.Tensor,
        logprob_b: torch.Tensor,
        adv_b: torch.Tensor,
        ret_b: torch.Tensor,
        val_b: torch.Tensor,
        epochs: int = 4,
        batch_size: int = 64
    ) -> Dict[str, float]:
        n_samples = obs_b.size(0)
        # Normalize advantages
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        total_loss, pg_loss, v_loss = 0.0, 0.0, 0.0
        updates = 0

        for _ in range(epochs):
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                idx = indices[start:end]

                _, new_logprob, entropy, new_value = self.ac.get_action_and_value(obs_b[idx], act_b[idx])
                logratio = new_logprob - logprob_b[idx]
                ratio = logratio.exp()

                # Policy loss
                mb_adv = adv_b[idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                mb_pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss with clipping
                v_loss_unclipped = (new_value - ret_b[idx]) ** 2
                v_clipped = val_b[idx] + torch.clamp(new_value - val_b[idx], -self.clip_coef, self.clip_coef)
                v_loss_clipped = (v_clipped - ret_b[idx]) ** 2
                mb_v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = mb_pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * mb_v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                pg_loss += mb_pg_loss.item()
                v_loss += mb_v_loss.item()
                updates += 1

        return {
            "loss": total_loss / max(updates, 1),
            "policy_loss": pg_loss / max(updates, 1),
            "value_loss": v_loss / max(updates, 1)
        }

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            dist = self.ac.actor(x)
            action = dist.mean if deterministic else dist.sample()
            action = torch.clamp(action, 0.0, 1.0)
            return action.squeeze(0).cpu().numpy()
