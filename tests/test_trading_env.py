import pytest
import numpy as np
from alpha_rl.data.generators import MultiAssetMarketDataGenerator
from alpha_rl.data.features import build_feature_matrix
from alpha_rl.envs.trading_env import MultiAssetTradingEnv

@pytest.fixture
def sample_env():
    gen = MultiAssetMarketDataGenerator(assets=["BTC", "ETH"], seed=42)
    mkt = gen.generate_ohlcv(n_bars=60)
    feats = {a: build_feature_matrix(df) for a, df in mkt.items()}
    return MultiAssetTradingEnv(mkt, feats, window_size=10)

def test_env_reset(sample_env):
    obs, info = sample_env.reset()
    assert obs.shape == sample_env.observation_space.shape
    assert sample_env.portfolio_value == 100_000.0

def test_env_step_invariants(sample_env):
    sample_env.reset()
    action = np.array([0.4, 0.4], dtype=np.float32)
    obs, reward, term, trunc, info = sample_env.step(action)
    assert "portfolio_value" in info
    assert sample_env.portfolio_value > 0.0
    assert sample_env.cash >= 0.0
    assert isinstance(reward, float)
