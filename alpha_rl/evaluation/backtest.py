import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Dict, Any

def purged_and_embargoed_kfold(
    n_samples: int,
    n_splits: int = 5,
    embargo_pct: float = 0.01
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Marcos Lopez de Prado's Purged and Embargoed K-Fold split.
    Prevents lookahead bias and serial correlation leakage in financial time series.
    """
    indices = np.arange(n_samples)
    embargo = int(n_samples * embargo_pct)
    test_fold_size = n_samples // n_splits

    for i in range(n_splits):
        test_start = i * test_fold_size
        test_end = (i + 1) * test_fold_size if i < n_splits - 1 else n_samples
        test_indices = indices[test_start:test_end]

        train_left = indices[:max(0, test_start)]
        train_right = indices[min(n_samples, test_end + embargo):]
        train_indices = np.concatenate([train_left, train_right])

        yield train_indices, test_indices

class BacktestEngine:
    """Vectorized Backtesting Engine & Comprehensive Quantitative Tearsheet Generator."""
    def __init__(self, risk_free_rate: float = 0.04, periods_per_year: int = 252):
        self.rf = risk_free_rate
        self.ppy = periods_per_year

    def compute_metrics(
        self,
        portfolio_values: np.ndarray,
        benchmark_values: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        n_periods = len(returns)
        if n_periods < 2:
            return {}

        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1.0
        cagr = (portfolio_values[-1] / portfolio_values[0]) ** (self.ppy / max(n_periods, 1)) - 1.0

        ann_mean = np.mean(returns) * self.ppy
        ann_vol = np.std(returns) * np.sqrt(self.ppy) + 1e-9
        sharpe = (ann_mean - self.rf) / ann_vol

        # Downside standard deviation (Sortino)
        downside_returns = returns[returns < 0.0]
        downside_std = np.std(downside_returns) * np.sqrt(self.ppy) if len(downside_returns) > 0 else 1e-9
        sortino = (ann_mean - self.rf) / max(downside_std, 1e-9)

        # Max Drawdown & Calmar
        cum_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (cum_max - portfolio_values) / cum_max
        max_dd = float(np.max(drawdowns))
        calmar = cagr / max(max_dd, 1e-4)

        # Win Rate & Profit Factor
        pos_rets = returns[returns > 0.0]
        neg_rets = returns[returns < 0.0]
        win_rate = len(pos_rets) / max(len(returns), 1)
        gross_profit = np.sum(pos_rets) if len(pos_rets) > 0 else 1e-9
        gross_loss = np.abs(np.sum(neg_rets)) if len(neg_rets) > 0 else 1e-9
        profit_factor = float(gross_profit / gross_loss)

        alpha, beta = 0.0, 1.0
        if benchmark_values is not None and len(benchmark_values) == len(portfolio_values):
            bench_returns = np.diff(benchmark_values) / benchmark_values[:-1]
            cov = np.cov(returns, bench_returns)[0, 1]
            var_b = np.var(bench_returns) + 1e-9
            beta = float(cov / var_b)
            alpha = float(ann_mean - (self.rf + beta * (np.mean(bench_returns) * self.ppy - self.rf)))

        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "alpha": float(alpha),
            "beta": float(beta)
        }
