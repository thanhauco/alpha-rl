import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

class MultiAssetTradingEnv(gym.Env):
    """
    Gymnasium multi-asset continuous portfolio allocation environment.
    Action: target portfolio weights across N assets (long-only simplex).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        market_data: Dict[str, pd.DataFrame],
        feature_data: Dict[str, pd.DataFrame],
        initial_balance: float = 100_000.0,
        window_size: int = 20,
        fee_rate: float = 0.0005,
    ):
        super().__init__()
        self.assets = sorted(list(market_data.keys()))
        self.n_assets = len(self.assets)
        self.market_data = market_data
        self.feature_data = feature_data
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.fee_rate = fee_rate

        self.n_features = feature_data[self.assets[0]].shape[1]
        self.n_bars = len(market_data[self.assets[0]])

        # Action: target weights for assets (cash is remainder)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

        # Obs: window of features + current weights + cash ratio
        obs_dim = (self.window_size * self.n_assets * self.n_features) + self.n_assets + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.current_step = 0
        self.portfolio_value = initial_balance
        self.cash = initial_balance
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)
        self.weights = np.zeros(self.n_assets, dtype=np.float64)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.cash = self.initial_balance
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)
        self.weights = np.zeros(self.n_assets, dtype=np.float64)
        return self._get_observation(), {}

    def _get_prices(self, step: int) -> np.ndarray:
        return np.array([self.market_data[a]["close"].iloc[step] for a in self.assets], dtype=np.float64)

    def _get_observation(self) -> np.ndarray:
        obs_list = []
        for a in self.assets:
            feat_window = self.feature_data[a].iloc[self.current_step - self.window_size:self.current_step].values
            obs_list.append(feat_window.flatten())
        
        flat_feats = np.concatenate(obs_list)
        cash_ratio = np.array([self.cash / max(self.portfolio_value, 1e-6)], dtype=np.float32)
        obs = np.concatenate([flat_feats, self.weights.astype(np.float32), cash_ratio])
        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        prices = self._get_prices(self.current_step)
        
        # Normalize action to simplex
        action = np.clip(action, 0.0, 1.0)
        total_weight = np.sum(action)
        if total_weight > 1.0:
            action = action / total_weight

        target_values = action * self.portfolio_value
        target_holdings = target_values / prices
        delta_holdings = target_holdings - self.holdings

        turnover = np.sum(np.abs(delta_holdings) * prices)
        fee = turnover * self.fee_rate

        self.holdings = target_holdings
        self.cash = self.portfolio_value - np.sum(self.holdings * prices) - fee

        self.current_step += 1
        terminated = self.current_step >= self.n_bars - 1
        truncated = False

        next_prices = self._get_prices(self.current_step)
        new_val = self.cash + np.sum(self.holdings * next_prices)
        reward = float(np.log(max(new_val, 1e-6) / max(self.portfolio_value, 1e-6)))

        self.portfolio_value = new_val
        self.weights = (self.holdings * next_prices) / max(self.portfolio_value, 1e-6)

        return self._get_observation(), reward, terminated, truncated, {
            "portfolio_value": self.portfolio_value,
            "turnover": turnover,
            "fee": fee
        }
