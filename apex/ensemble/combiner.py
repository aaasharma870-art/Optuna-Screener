"""Multi-strategy ensemble combiner.

Runs each strategy, computes per-strategy returns, derives risk-parity weights,
applies regime overlay, sums weighted positions to produce final portfolio NAV.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from apex.ensemble.risk_parity import compute_risk_parity_weights
from apex.ensemble.regime_overlay import apply_regime_tilts


class EnsembleCombiner:
    """Run a basket of strategies and combine via risk parity + regime overlay."""

    def __init__(self, strategies: List[Any],
                 max_weight: float = 0.30,
                 vol_lookback_days: int = 60,
                 size_change_threshold: float = 0.10):
        self.strategies = strategies
        self.max_weight = max_weight
        self.vol_lookback_days = vol_lookback_days
        self.size_change_threshold = size_change_threshold

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute every strategy and combine.

        Returns:
          {
            'per_strategy_signals': dict[name -> signals DataFrame],
            'per_strategy_positions': dict[name -> position Series],
            'weights': dict[name -> final weight],
            'portfolio_position': Series (combined position over time),
            'trades': list[dict] (rebalance events),
          }

        Note: strategies named "cross_asset_vol_overlay" are treated specially:
        their compute_position_size output is a per-bar SIZE MULTIPLIER applied
        to the combined portfolio position, NOT summed as a directional signal.
        """
        per_strategy_signals: Dict[str, pd.DataFrame] = {}
        per_strategy_positions: Dict[str, pd.Series] = {}
        per_strategy_returns: Dict[str, pd.Series] = {}

        # Determine length n for the overlay multiplier (default 1.0)
        exec_df = data.get("exec_df_1H", pd.DataFrame())
        n_bars = len(exec_df)
        overlay_mult = pd.Series([1.0] * n_bars)

        # Track overlay names so they are NOT summed into the directional
        # combined position and NOT included in the risk-parity weighting.
        overlay_names: List[str] = []

        non_overlay_strategies = []
        for s in self.strategies:
            if s.name == "cross_asset_vol_overlay":
                sig = s.compute_signals(data)
                overlay_mult = s.compute_position_size(data, sig)
                # Track overlay's signals/positions for transparency
                per_strategy_signals[s.name] = sig
                per_strategy_positions[s.name] = overlay_mult
                overlay_names.append(s.name)
            else:
                non_overlay_strategies.append(s)

        for s in non_overlay_strategies:
            sig = s.compute_signals(data)
            pos = s.compute_position_size(data, sig)
            per_strategy_signals[s.name] = sig
            per_strategy_positions[s.name] = pos
            # Approximate per-strategy returns for vol estimation:
            # change in position * subsequent price change.
            close = data.get("exec_df_1H", pd.DataFrame()).get("close")
            if close is not None and len(close) > 1:
                price_returns = close.pct_change().fillna(0.0).values
                strategy_returns = pos.shift(1).fillna(0.0).values * price_returns
                per_strategy_returns[s.name] = pd.Series(
                    strategy_returns, index=pos.index)
            else:
                per_strategy_returns[s.name] = pd.Series([0.0] * len(pos))

        # Risk-parity weights from rolling vol of per-strategy returns
        weights = compute_risk_parity_weights(
            per_strategy_returns,
            lookback_days=self.vol_lookback_days,
            max_weight=self.max_weight,
        )

        # Regime overlay: use the dominant regime in the data window
        regime_series = data.get("regime_state")
        if regime_series is not None and len(regime_series) > 0:
            mode = regime_series.dropna().mode()
            current_regime = mode.iloc[0] if len(mode) > 0 else "UNKNOWN"
        else:
            current_regime = "UNKNOWN"
        weights = apply_regime_tilts(weights, current_regime)

        # Combine per-strategy positions (exclude overlay strategies — they
        # contribute as a final-stage size multiplier, not a directional bet).
        n = len(next(iter(per_strategy_positions.values())))
        combined = pd.Series([0.0] * n)
        for name, pos in per_strategy_positions.items():
            if name in overlay_names:
                continue
            w = weights.get(name, 0.0)
            combined = combined + w * pos.values

        # Apply overlay multiplier (per-bar size scaler) to combined position
        if len(overlay_mult) == n:
            combined = combined * overlay_mult.values

        # Generate "trade" events whenever combined position shifts > threshold
        trades = []
        prev_pos = 0.0
        for i, p in enumerate(combined):
            if abs(p - prev_pos) >= self.size_change_threshold:
                trades.append({
                    "bar_idx": i,
                    "old_position": float(prev_pos),
                    "new_position": float(p),
                    "delta": float(p - prev_pos),
                })
                prev_pos = p

        return {
            "per_strategy_signals": per_strategy_signals,
            "per_strategy_positions": per_strategy_positions,
            "per_strategy_returns": per_strategy_returns,
            "weights": weights,
            "portfolio_position": combined,
            "trades": trades,
            "current_regime": current_regime,
        }
