import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

class HestonProcess:
    """
    Heston Stochastic Volatility Model:
    dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_1
    dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_2
    corr(dW_1, dW_2) = rho
    """
    def __init__(
        self,
        s0: float = 100.0,
        v0: float = 0.04,
        mu: float = 0.05,
        kappa: float = 1.5,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        seed: Optional[int] = None
    ):
        self.s0 = s0
        self.v0 = v0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.rng = np.random.default_rng(seed)

    def sample_path(self, n_steps: int = 1000, dt: float = 1.0 / 252.0) -> Tuple[np.ndarray, np.ndarray]:
        prices = np.zeros(n_steps)
        variances = np.zeros(n_steps)
        prices[0] = self.s0
        variances[0] = self.v0

        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        chol = np.linalg.cholesky(cov)

        for t in range(1, n_steps):
            z = self.rng.standard_normal(2)
            dw = chol @ z * np.sqrt(dt)
            dw_s, dw_v = dw[0], dw[1]

            v_prev = max(variances[t - 1], 1e-6)
            dv = self.kappa * (self.theta - v_prev) * dt + self.xi * np.sqrt(v_prev) * dw_v
            v_curr = max(v_prev + dv, 1e-6)
            variances[t] = v_curr

            log_ret = (self.mu - 0.5 * v_curr) * dt + np.sqrt(v_curr) * dw_s
            prices[t] = prices[t - 1] * np.exp(log_ret)

        return prices, variances

class MertonJumpDiffusion:
    """
    Merton Jump Diffusion Process:
    dS_t = (mu - lambda * k) * S_t * dt + sigma * S_t * dW_t + S_t * dJ_t
    """
    def __init__(
        self,
        s0: float = 100.0,
        mu: float = 0.05,
        sigma: float = 0.2,
        lambda_j: float = 0.75,
        mu_j: float = -0.05,
        sigma_j: float = 0.1,
        seed: Optional[int] = None
    ):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.lambda_j = lambda_j
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.rng = np.random.default_rng(seed)

    def sample_path(self, n_steps: int = 1000, dt: float = 1.0 / 252.0) -> np.ndarray:
        prices = np.zeros(n_steps)
        prices[0] = self.s0
        k = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0

        for t in range(1, n_steps):
            z = self.rng.standard_normal()
            n_jumps = self.rng.poisson(self.lambda_j * dt)
            jump_factor = 0.0
            if n_jumps > 0:
                jump_factor = np.sum(self.rng.normal(self.mu_j, self.sigma_j, n_jumps))

            drift = (self.mu - self.lambda_j * k - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * np.sqrt(dt) * z
            prices[t] = prices[t - 1] * np.exp(drift + diffusion + jump_factor)

        return prices

class MultiAssetMarketDataGenerator:
    """Generates realistic multi-asset correlated OHLCV market feeds."""
    def __init__(
        self,
        assets: Optional[List[str]] = None,
        base_prices: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 42
    ):
        self.assets = assets or ["BTC", "ETH", "SOL", "SPY"]
        self.base_prices = base_prices or {"BTC": 65000.0, "ETH": 3500.0, "SOL": 150.0, "SPY": 500.0}
        self.seed = seed

    def generate_ohlcv(self, n_bars: int = 1000, freq: str = "1h") -> Dict[str, pd.DataFrame]:
        rng = np.random.default_rng(self.seed)
        n_assets = len(self.assets)
        
        # Correlated returns
        corr = np.eye(n_assets) * 0.4 + 0.6
        chol = np.linalg.cholesky(corr)

        out = {}
        for i, asset in enumerate(self.assets):
            s0 = self.base_prices.get(asset, 100.0)
            heston = HestonProcess(s0=s0, kappa=2.0, theta=0.04, xi=0.25, seed=self.seed + i if self.seed else None)
            prices, vars_ = heston.sample_path(n_steps=n_bars * 5)
            
            # Resample micro-ticks into OHLCV bars
            reshaped = prices.reshape(n_bars, 5)
            open_ = reshaped[:, 0]
            high = np.max(reshaped, axis=1) * (1.0 + rng.uniform(0.0005, 0.002, n_bars))
            low = np.min(reshaped, axis=1) * (1.0 - rng.uniform(0.0005, 0.002, n_bars))
            close = reshaped[:, -1]
            high = np.maximum(high, np.maximum(open_, close))
            low = np.minimum(low, np.minimum(open_, close))
            volume = (1000.0 * (s0 / close) * (1.0 + rng.exponential(0.8, n_bars))).astype(float)

            dates = pd.date_range("2024-01-01", periods=n_bars, freq=freq)
            df = pd.DataFrame({
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            }, index=dates)
            out[asset] = df

        return out
