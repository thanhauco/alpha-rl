import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

class MultiAssetTradingEnv(gym.Env):
    """
    Gymnasium multi-asset continuous portfolio allocation environment.
    Supports continuous action space with softmax simplex projection and optional cash holding.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        market_data: Dict[str, pd.DataFrame],
        feature_data: Dict[str, pd.DataFrame],
        initial_balance: float = 100_000.0,
        window_size: int = 20,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0005,
        slippage_lambda: float = 1e-7,
        cash_buffer_pct: float = 0.01,
        allow_short: bool = False
    ):
        super().__init__()
        self.assets = sorted(list(market_data.keys()))
        self.n_assets = len(self.assets)
        self.market_data = market_data
        self.feature_data = feature_data
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_lambda = slippage_lambda
        self.cash_buffer_pct = cash_buffer_pct
        self.allow_short = allow_short

        self.n_features = feature_data[self.assets[0]].shape[1]
        self.n_bars = len(market_data[self.assets[0]])

        # Action: N asset allocations (+1 implicit for cash)
        self.action_space = spaces.Box(low=-1.0 if allow_short else 0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
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

    def _get_volumes(self, step: int) -> np.ndarray:
        return np.array([self.market_data[a]["volume"].iloc[step] for a in self.assets], dtype=np.float64)

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
        volumes = self._get_volumes(self.current_step)
        
        if not self.allow_short:
            action = np.clip(action, 0.0, 1.0)
            max_alloc = 1.0 - self.cash_buffer_pct
            total_weight = np.sum(action)
            if total_weight > max_alloc:
                action = action * (max_alloc / total_weight)
        else:
            action = np.clip(action, -1.0, 1.0)
            total_abs = np.sum(np.abs(action))
            if total_abs > 1.0:
                action = action / total_abs

        target_values = action * self.portfolio_value
        target_holdings = target_values / prices
        delta_holdings = target_holdings - self.holdings
        trade_sizes = np.abs(delta_holdings) * prices

        slippage_cost = np.sum(self.slippage_lambda * (trade_sizes ** 2) / (volumes + 1e-6))
        fee = np.sum(trade_sizes * self.taker_fee)

        self.holdings = target_holdings
        self.cash = max(0.0, self.portfolio_value - np.sum(self.holdings * prices) - fee - slippage_cost)

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
            "turnover": float(np.sum(trade_sizes)),
            "fee": float(fee),
            "slippage": float(slippage_cost)
        }
