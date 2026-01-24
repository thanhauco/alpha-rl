import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

class MultiAssetTradingEnv(gym.Env):
    """
    Gymnasium multi-asset continuous portfolio allocation environment.
    Supports Differential Sharpe Ratio (Moody & Saffell) and Downside Sortino reward modes.
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
        reward_mode: str = "differential_sharpe",
        eta_sharpe: float = 0.05
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
        self.reward_mode = reward_mode
        self.eta_sharpe = eta_sharpe

        # Exponential moving averages for Differential Sharpe Ratio
        self.a_t = 0.0
        self.b_t = 0.0

        self.n_features = feature_data[self.assets[0]].shape[1]
        self.n_bars = len(market_data[self.assets[0]])

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
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
        self.a_t = 0.0
        self.b_t = 0.0
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
        
        action = np.clip(action, 0.0, 1.0)
        max_alloc = 1.0 - self.cash_buffer_pct
        total_weight = np.sum(action)
        if total_weight > max_alloc:
            action = action * (max_alloc / total_weight)

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
        simple_ret = (new_val - self.portfolio_value) / max(self.portfolio_value, 1e-6)

        if self.reward_mode == "differential_sharpe":
            # Differential Sharpe Ratio formulation (Moody & Saffell 1998)
            delta_a = simple_ret - self.a_t
            delta_b = (simple_ret ** 2) - self.b_t
            var_term = max(self.b_t - (self.a_t ** 2), 1e-6)
            dsr = (self.b_t * delta_a - 0.5 * self.a_t * delta_b) / (var_term ** 1.5)
            reward = float(np.clip(dsr, -5.0, 5.0))
            self.a_t += self.eta_sharpe * delta_a
            self.b_t += self.eta_sharpe * delta_b
        elif self.reward_mode == "downside_sortino":
            downside = min(0.0, simple_ret) ** 2
            reward = float(simple_ret - 2.0 * downside)
        else:
            reward = float(np.log(max(new_val, 1e-6) / max(self.portfolio_value, 1e-6)))

        self.portfolio_value = new_val
        self.weights = (self.holdings * next_prices) / max(self.portfolio_value, 1e-6)

        return self._get_observation(), reward, terminated, truncated, {
            "portfolio_value": self.portfolio_value,
            "turnover": float(np.sum(trade_sizes)),
            "fee": float(fee),
            "slippage": float(slippage_cost)
        }
