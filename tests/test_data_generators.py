import pytest
import numpy as np
from alpha_rl.data.generators import HestonProcess, MertonJumpDiffusion, MultiAssetMarketDataGenerator

def test_heston_process_sampling():
    heston = HestonProcess(s0=100.0, seed=42)
    prices, variances = heston.sample_path(n_steps=100)
    assert len(prices) == 100
    assert len(variances) == 100
    assert np.all(prices > 0.0)
    assert np.all(variances > 0.0)

def test_merton_jump_diffusion_sampling():
    merton = MertonJumpDiffusion(s0=100.0, seed=42)
    prices = merton.sample_path(n_steps=100)
    assert len(prices) == 100
    assert np.all(prices > 0.0)

def test_multi_asset_generator():
    gen = MultiAssetMarketDataGenerator(assets=["BTC", "ETH"], seed=42)
    data = gen.generate_ohlcv(n_bars=50)
    assert "BTC" in data
    assert "ETH" in data
    df_btc = data["BTC"]
    assert len(df_btc) == 50
    assert list(df_btc.columns) == ["open", "high", "low", "close", "volume"]
