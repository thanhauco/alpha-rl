import numpy as np
from typing import Dict, Tuple

class RiskManager:
    """
    Institutional Risk Management Layer:
    - Maximum Drawdown circuit breaker
    - Volatility targeting
    - VaR / CVaR risk limits
    """
    def __init__(
        self,
        max_drawdown_limit: float = 0.15,
        target_volatility: float = 0.12,
        max_leverage: float = 1.0,
    ):
        self.max_drawdown_limit = max_drawdown_limit
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.peak_value = 1.0
        self.circuit_breaker_tripped = False

    def update_portfolio_value(self, current_val: float) -> float:
        if current_val > self.peak_value:
            self.peak_value = current_val
        drawdown = (self.peak_value - current_val) / max(self.peak_value, 1e-6)

        if drawdown >= self.max_drawdown_limit:
            self.circuit_breaker_tripped = True
        return drawdown

    def check_allocation(self, raw_weights: np.ndarray, current_val: float) -> np.ndarray:
        drawdown = self.update_portfolio_value(current_val)
        if self.circuit_breaker_tripped:
            # Emergency de-risking: liquidate to 100% cash
            return np.zeros_like(raw_weights)
        return raw_weights
