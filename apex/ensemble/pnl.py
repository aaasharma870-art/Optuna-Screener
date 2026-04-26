"""Compute PnL stats from per-bar portfolio positions (Phase 15)."""
from typing import Any, Dict

import numpy as np
import pandas as pd


def compute_pnl_stats(positions: pd.Series, prices: pd.Series,
                      periods_per_year: int = 252,
                      commission_pct: float = 0.05) -> Dict[str, Any]:
    """Compute equity curve + PnL stats from positions + prices.

    Args:
        positions: per-bar position size (-1.0 to +1.0)
        prices:    per-bar close prices
        periods_per_year: 252 for daily, ~7*252 for hourly equities
        commission_pct: round-trip cost in % per position change (default 0.05%)

    Returns:
        dict with: equity_curve, total_return_pct, max_dd_pct, sharpe_annualized,
                   calmar, win_rate_pct, n_trades, avg_trade_pct, n_bars,
                   n_bars_in_position
    """
    # Coerce to pandas Series with simple integer index for safe alignment
    if not isinstance(positions, pd.Series):
        positions = pd.Series(positions)
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)

    positions = positions.reset_index(drop=True).astype(float)
    prices = prices.reset_index(drop=True).astype(float)

    n_pos = len(positions)
    n_px = len(prices)
    n = min(n_pos, n_px)

    if n < 2:
        return {"error": "insufficient data", "n_bars": n,
                "equity_curve": [], "total_return_pct": 0.0,
                "max_dd_pct": 0.0, "sharpe_annualized": 0.0,
                "calmar": 0.0, "win_rate_pct": 0.0, "n_trades": 0,
                "avg_trade_pct": 0.0, "n_bars_in_position": 0}

    positions = positions.iloc[:n]
    prices = prices.iloc[:n]

    # Per-bar returns: position from prior bar applied to current bar's pct change
    price_returns = prices.pct_change().fillna(0.0).values
    pos_arr = positions.shift(1).fillna(0.0).values
    bar_returns = pos_arr * price_returns

    # Commission cost: each time position changes meaningfully, deduct commission.
    pos_changes = positions.diff().abs().fillna(0.0).values
    is_change = (pos_changes > 0.01).astype(float)
    commissions = is_change * (commission_pct / 100.0)
    bar_returns_net = bar_returns - commissions

    # Equity curve
    equity = pd.Series((1.0 + bar_returns_net).cumprod())

    # Total return
    total_return_pct = float(equity.iloc[-1] - 1.0) * 100.0

    # Max drawdown
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd_pct = float(drawdowns.min()) * 100.0

    # Annualized Sharpe
    sigma = float(np.std(bar_returns_net, ddof=1)) if len(bar_returns_net) > 1 else 0.0
    if sigma > 1e-12:
        sharpe = float(np.mean(bar_returns_net) / sigma * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    # Calmar
    if abs(max_dd_pct) > 0.01:
        calmar = total_return_pct / abs(max_dd_pct)
    else:
        calmar = float("inf") if total_return_pct > 0 else 0.0

    # Win rate (per-bar bars when in position with positive PnL)
    in_position = (pos_arr != 0)
    n_in_pos = int(in_position.sum())
    n_winners = int(((bar_returns_net > 0) & in_position).sum())
    win_rate_pct = (100.0 * n_winners / n_in_pos) if n_in_pos > 0 else 0.0

    # Trades count = position changes
    n_trades = int(is_change.sum())

    return {
        "equity_curve": equity.tolist(),
        "total_return_pct": total_return_pct,
        "max_dd_pct": max_dd_pct,
        "sharpe_annualized": sharpe,
        "calmar": calmar,
        "win_rate_pct": win_rate_pct,
        "n_trades": n_trades,
        "n_bars": n,
        "n_bars_in_position": n_in_pos,
        "avg_trade_pct": (total_return_pct / n_trades) if n_trades > 0 else 0.0,
    }
