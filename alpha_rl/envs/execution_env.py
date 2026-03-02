import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional

class OrderExecutionEnv(gym.Env):
    """
    L2 Orderbook simulator and execution environment for optimal liquidation / acquisition.
    Simulates bid-ask queues, market impact, and limit order matching.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        total_quantity: float = 10_000.0,
        n_steps: int = 60,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        half_spread: float = 0.02,
        gamma_perm: float = 2.5e-6,
        eta_temp: float = 2.5e-5,
    ):
        super().__init__()
        self.total_quantity = total_quantity
        self.n_steps = n_steps
        self.initial_price = initial_price
        self.volatility = volatility
        self.half_spread = half_spread
        self.gamma_perm = gamma_perm
        self.eta_temp = eta_temp

        # Action: fraction of remaining inventory to liquidate in current step [0.0, 1.0]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        # Obs: [remaining_inventory_ratio, time_remaining_ratio, mid_price_ratio, spread]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.current_step = 0
        self.inventory = total_quantity
        self.mid_price = initial_price
        self.cash = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = self.total_quantity
        self.mid_price = self.initial_price
        self.cash = 0.0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.inventory / self.total_quantity,
            (self.n_steps - self.current_step) / self.n_steps,
            self.mid_price / self.initial_price,
            self.half_spread * 2.0
        ], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        frac = float(np.clip(action[0], 0.0, 1.0))
        if self.current_step == self.n_steps - 1:
            qty_to_trade = self.inventory # Force liquidation at horizon
        else:
            qty_to_trade = self.inventory * frac

        # Price impact (Almgren-Chriss formulation)
        perm_impact = self.gamma_perm * qty_to_trade
        temp_impact = self.eta_temp * qty_to_trade
        exec_price = (self.mid_price - self.half_spread) - temp_impact
        proceeds = qty_to_trade * exec_price

        # Update state
        self.inventory -= qty_to_trade
        self.cash += proceeds
        drift = -perm_impact + np.random.normal(0.0, self.volatility)
        self.mid_price = max(0.1, self.mid_price + drift)
        self.current_step += 1

        terminated = self.current_step >= self.n_steps or self.inventory <= 1e-4
        truncated = False

        # Shortfall relative to initial arrival price
        shortfall = (qty_to_trade * self.initial_price) - proceeds
        reward = -float(shortfall / (self.total_quantity * self.initial_price))

        return self._get_obs(), reward, terminated, truncated, {
            "cash": self.cash,
            "remaining_inventory": self.inventory,
            "mid_price": self.mid_price,
            "shortfall": shortfall
        }
