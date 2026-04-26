"""Walk-forward validation of ensemble weights."""
from typing import Any, Dict

import numpy as np
import pandas as pd

from apex.ensemble.risk_parity import compute_risk_parity_weights


def compare_dynamic_vs_static_weights(
        monthly_returns: Dict[str, pd.Series],
        warmup_months: int = 6,
) -> Dict[str, Any]:
    """Compare ensemble Sharpe under two weight regimes:
      - Static: weights computed once from first `warmup_months`, never updated
      - Dynamic: weights re-computed each month from trailing 12-month window

    Returns {'static_sharpe', 'dynamic_sharpe', 'uplift', 'n_months'}.
    """
    if not monthly_returns:
        return {"error": "empty returns dict", "n_months": 0}
    n_months = min(len(r) for r in monthly_returns.values())
    if n_months < warmup_months + 3:
        return {"error": "insufficient history", "n_months": n_months}

    strategy_names = list(monthly_returns.keys())

    # Static weights from warmup window
    warmup = {n: monthly_returns[n].iloc[:warmup_months] for n in strategy_names}
    static_weights = compute_risk_parity_weights(warmup, lookback_days=warmup_months,
                                                  max_weight=0.30)

    static_returns = []
    dynamic_returns = []
    for m in range(warmup_months, n_months):
        # Static: use warmup weights
        s_ret = sum(static_weights.get(n, 0.0) * monthly_returns[n].iloc[m]
                    for n in strategy_names)
        static_returns.append(s_ret)

        # Dynamic: recompute from trailing 12-month window (or warmup_months min)
        lookback_start = max(0, m - 12)
        recent = {n: monthly_returns[n].iloc[lookback_start:m]
                  for n in strategy_names}
        dyn_weights = compute_risk_parity_weights(recent,
                                                   lookback_days=12,
                                                   max_weight=0.30)
        d_ret = sum(dyn_weights.get(n, 0.0) * monthly_returns[n].iloc[m]
                    for n in strategy_names)
        dynamic_returns.append(d_ret)

    static_arr = np.array(static_returns)
    dynamic_arr = np.array(dynamic_returns)

    def _sharpe(arr):
        if len(arr) < 2 or arr.std() <= 1e-12:
            return 0.0
        return float(arr.mean() / arr.std(ddof=1) * np.sqrt(12))

    static_sharpe = _sharpe(static_arr)
    dynamic_sharpe = _sharpe(dynamic_arr)

    return {
        "static_sharpe": static_sharpe,
        "dynamic_sharpe": dynamic_sharpe,
        "uplift": dynamic_sharpe - static_sharpe,
        "n_months": n_months,
    }
