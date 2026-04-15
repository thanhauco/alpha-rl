import numpy as np
from typing import Dict, Tuple

class RiskManager:
    """
    Institutional Risk Management Layer:
    - Maximum Drawdown circuit breaker
    - Volatility targeting with robust epsilon guards
    - Value-at-Risk (VaR 95%, 99%) & Conditional VaR (Expected Shortfall)
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

    def calculate_volatility_scaler(self, realized_vol: float) -> float:
        # Guard against zero or near-zero volatility to avoid division overflow
        safe_vol = max(realized_vol, 1e-4)
        scaler = self.target_volatility / safe_vol
        return float(np.clip(scaler, 0.1, self.max_leverage))

    def calculate_var_cvar(self, returns: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
        if len(returns) < 10:
            return 0.0, 0.0
        sorted_rets = np.sort(returns)
        idx = max(1, int(alpha * len(sorted_rets)))
        var = -sorted_rets[idx]
        cvar = -np.mean(sorted_rets[:idx])
        return float(var), float(cvar)

    def check_allocation(self, raw_weights: np.ndarray, current_val: float, realized_vol: float = 0.12) -> np.ndarray:
        drawdown = self.update_portfolio_value(current_val)
        if self.circuit_breaker_tripped:
            return np.zeros_like(raw_weights)
        
        scaler = self.calculate_volatility_scaler(realized_vol)
        scaled_weights = raw_weights * scaler
        total_w = np.sum(np.abs(scaled_weights))
        if total_w > 1.0:
            scaled_weights = scaled_weights / total_w
        return scaled_weights
