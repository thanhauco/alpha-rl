import pytest
import torch
import numpy as np
from alpha_rl.agents.ppo import PPOAgent

def test_ppo_prediction():
    agent = PPOAgent(obs_dim=32, action_dim=4)
    obs = np.random.randn(32).astype(np.float32)
    action = agent.predict(obs, deterministic=True)
    assert action.shape == (4,)
    assert np.all(action >= 0.0)
    assert np.all(action <= 1.0)

def test_ppo_gae_and_update():
    agent = PPOAgent(obs_dim=16, action_dim=2)
    rewards = np.array([0.1, -0.05, 0.2, 0.0], dtype=np.float32)
    values = np.array([0.05, 0.02, 0.1, 0.0], dtype=np.float32)
    dones = np.array([0, 0, 0, 1], dtype=np.float32)
    advs, rets = agent.compute_gae(rewards, values, dones, next_value=0.0)
    assert len(advs) == 4
    assert len(rets) == 4

    # Dummy batch update
    obs_b = torch.randn(8, 16)
    act_b = torch.rand(8, 2)
    logprob_b = torch.zeros(8)
    adv_b = torch.randn(8)
    ret_b = torch.randn(8)
    val_b = torch.randn(8)

    losses = agent.update(obs_b, act_b, logprob_b, adv_b, ret_b, val_b, epochs=1, batch_size=4)
    assert "loss" in losses
    assert "policy_loss" in losses
