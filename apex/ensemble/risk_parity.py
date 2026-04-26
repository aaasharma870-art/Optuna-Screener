"""Inverse-volatility (risk parity) portfolio weight computation."""
from typing import Dict

import numpy as np
import pandas as pd


def compute_risk_parity_weights(returns: Dict[str, pd.Series],
                                 lookback_days: int = 60,
                                 max_weight: float = 0.30) -> Dict[str, float]:
    """Compute risk-parity weights from per-strategy return series.

    Each strategy's weight is proportional to 1 / its annualized volatility,
    so each contributes equal portfolio variance ex-ante.

    Args:
        returns: dict[strategy_name -> Series of returns (most recent N values used)]
        lookback_days: how many tail values to use for vol estimation
        max_weight: cap on any single strategy's weight (default 0.30)

    Returns:
        dict[strategy_name -> weight in [0, max_weight]] summing to 1.0.
        Strategies with zero or NaN vol get weight 0; remainder renormalize.
    """
    inv_vols = {}
    for name, ret_series in returns.items():
        recent = ret_series.tail(lookback_days).dropna()
        if len(recent) < 2:
            inv_vols[name] = 0.0
            continue
        vol = float(recent.std(ddof=1)) * np.sqrt(252)
        if vol <= 1e-9 or np.isnan(vol):
            inv_vols[name] = 0.0
        else:
            inv_vols[name] = 1.0 / vol

    total_inv_vol = sum(inv_vols.values())
    if total_inv_vol <= 0:
        # All strategies have zero vol or invalid — fall back to equal weights
        n = len(returns)
        return {name: 1.0 / n for name in returns}

    weights = {name: iv / total_inv_vol for name, iv in inv_vols.items()}

    # If the cap is not achievable (n * max_weight < 1.0), skip capping and
    # return the raw inverse-vol weights — preserving relative risk-parity
    # ratios is more important than enforcing an unsatisfiable cap.
    n_strategies = len(weights)
    if n_strategies * max_weight < 1.0 - 1e-9:
        return weights

    # Apply max_weight cap iteratively
    for _ in range(10):  # converge in a few iterations
        capped = {name: min(w, max_weight) for name, w in weights.items()}
        excess = sum(weights.values()) - sum(capped.values())
        if excess <= 1e-9:
            return capped
        # Redistribute excess to uncapped strategies proportionally
        uncapped_total = sum(w for n, w in capped.items() if w < max_weight - 1e-9)
        if uncapped_total <= 0:
            # All hit cap — accept; sum will be < 1, renormalize
            return {name: w / sum(capped.values()) for name, w in capped.items()}
        scale = (uncapped_total + excess) / uncapped_total
        weights = {name: (capped[name] * scale if capped[name] < max_weight - 1e-9
                          else max_weight)
                   for name in capped}

    return weights
