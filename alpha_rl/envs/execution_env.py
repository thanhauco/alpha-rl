import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional

class AlmgrenChrissBenchmark:
    """
    Analytical benchmark solver for Almgren & Chriss (2000) optimal execution.
    Solves for the optimal trajectory minimizing expected shortfall and variance.
    """
    def __init__(self, total_quantity: float, n_steps: int, gamma: float, eta: float, sigma: float, risk_aversion: float = 1e-5):
        self.X = total_quantity
        self.N = n_steps
        self.gamma = gamma
        self.eta = eta
        self.sigma = sigma
        self.lmbda = risk_aversion
        self.tau = 1.0

        # Kappa parameter
        kappa_sq = (self.lmbda * (self.sigma ** 2)) / (self.eta * (1.0 - 0.5 * self.gamma * self.tau))
        self.kappa = np.sqrt(max(kappa_sq, 1e-9))

    def get_trajectory(self) -> np.ndarray:
        t = np.arange(self.N + 1)
        # hyperbolic trajectory
        num = np.sinh(self.kappa * (self.N - t))
        denom = np.sinh(self.kappa * self.N)
        return self.X * (num / max(denom, 1e-9))

    def get_trade_sizes(self) -> np.ndarray:
        traj = self.get_trajectory()
        return np.diff(traj) * -1.0

class OrderExecutionEnv(gym.Env):
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

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
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
            qty_to_trade = self.inventory
        else:
            qty_to_trade = self.inventory * frac

        perm_impact = self.gamma_perm * qty_to_trade
        temp_impact = self.eta_temp * qty_to_trade
        exec_price = (self.mid_price - self.half_spread) - temp_impact
        proceeds = qty_to_trade * exec_price

        self.inventory -= qty_to_trade
        self.cash += proceeds
        drift = -perm_impact + np.random.normal(0.0, self.volatility)
        self.mid_price = max(0.1, self.mid_price + drift)
        self.current_step += 1

        terminated = self.current_step >= self.n_steps or self.inventory <= 1e-4
        truncated = False

        shortfall = (qty_to_trade * self.initial_price) - proceeds
        reward = -float(shortfall / (self.total_quantity * self.initial_price))

        return self._get_obs(), reward, terminated, truncated, {
            "cash": self.cash,
            "remaining_inventory": self.inventory,
            "mid_price": self.mid_price,
            "shortfall": shortfall
        }
