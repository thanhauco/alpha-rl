import pytest
import numpy as np
from alpha_rl.risk.manager import RiskManager

def test_risk_manager_drawdown_circuit_breaker():
    rm = RiskManager(max_drawdown_limit=0.10)
    rm.update_portfolio_value(100.0)
    w = np.array([0.5, 0.5])
    # Mild drop
    w_out = rm.check_allocation(w, 95.0)
    assert not rm.circuit_breaker_tripped
    assert np.sum(w_out) > 0.0

    # Breach 10% drawdown
    w_out_emergency = rm.check_allocation(w, 88.0)
    assert rm.circuit_breaker_tripped
    assert np.all(w_out_emergency == 0.0)

def test_volatility_targeting_near_zero():
    rm = RiskManager(target_volatility=0.15)
    scaler = rm.calculate_volatility_scaler(0.0)
    assert scaler == 1.0
    scaler_high = rm.calculate_volatility_scaler(0.30)
    assert scaler_high == 0.5
