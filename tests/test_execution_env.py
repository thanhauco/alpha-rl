import pytest
import numpy as np
from alpha_rl.envs.execution_env import OrderExecutionEnv, AlmgrenChrissBenchmark

def test_almgren_chriss_analytical_solution():
    bench = AlmgrenChrissBenchmark(total_quantity=1000.0, n_steps=10, gamma=1e-5, eta=1e-4, sigma=0.01)
    traj = bench.get_trajectory()
    trades = bench.get_trade_sizes()
    assert len(traj) == 11
    assert np.isclose(traj[0], 1000.0)
    assert np.isclose(traj[-1], 0.0, atol=1e-3)
    assert np.isclose(np.sum(trades), 1000.0, atol=1e-3)

def test_order_execution_env_liquidation():
    env = OrderExecutionEnv(total_quantity=5000.0, n_steps=20)
    obs, _ = env.reset()
    assert obs[0] == 1.0 # 100% inventory

    for _ in range(20):
        obs, reward, term, trunc, info = env.step(np.array([0.1]))
        if term:
            break
    assert env.inventory <= 1e-3
    assert env.cash > 0.0
