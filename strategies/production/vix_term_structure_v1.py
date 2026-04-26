"""
VIX Term Structure v1 — Production-Validated Strategy Preset
=============================================================

Deployable preset of the VIX/VIX3M term-structure mean-reversion strategy,
with parameters tuned via Phase 13 Optuna CPCV search and validated via
Phase 15 honest 75/25 holdout split (tuner never saw the holdout window).

Validation date:  2026-04-26
Pipeline version: Phase 16 (post-ensemble overhaul)

VALIDATION RESULTS (heavy budget, 150 Optuna trials, CPCV-tuned)
----------------------------------------------------------------

Backtest universe: SPY + QQQ
Data window:       2021-01-01 → 2026-04-26 (~5.3 years 1H bars per symbol)
Tune split:        75% (7530 bars/sym, 2021-Q1 → mid-2025)
Holdout split:     25% (2510 bars/sym, mid-2025 → 2026-Q1) — TRUE OOS

| Metric                         | TUNE       | HOLDOUT (TRUE OOS) |
|--------------------------------|------------|--------------------|
| Layer B CPCV median Sharpe     | 5.00       | 3.85               |
| Layer B CPCV % positive folds  | 100% (28)  | 100% (28)          |
| Trades                         | 345        | 108                |
| Win rate                       | 58.0%      | 58.4%              |
| Total return                   | +280.94%   | +37.68%            |
| Max drawdown                   | -3.91%     | -3.00%             |
| Sharpe (annualized)            | 4.26       | 3.27               |
| Calmar (return/abs(MaxDD))     | 71.80      | 12.58              |

Annualized holdout return: ~21% on unseen data with -3% MaxDD.

DECAY: tune Sharpe 4.26 → holdout Sharpe 3.27 (-23%). Real overfitting impact
exists but the strategy still clears Sharpe 3 OOS.

STRATEGY MECHANICS (from spec §5.3)
------------------------------------

Compute `ts_ratio = VIX / VIX3M` daily; forward-fill to 1H bars.
Compute 5-bar RSI on `ts_ratio` itself for confirmation.

LONG SPY when:
  - ts_ratio < contango_extreme_threshold (0.918) → vol curve overpriced near term
  - 5-bar RSI on ts_ratio < 30 (oversold confirmation)

SHORT SPY when:
  - ts_ratio > backwardation_extreme_threshold (1.141) → spike, expect reversal
  - 5-bar RSI on ts_ratio > 70 (overbought confirmation)

EXIT:
  - ts_ratio reverts to neutral band [0.946, 1.000]
  - OR ATR-stop = entry ± 1.643 × ATR
  - OR time stop = 8 bars (~1.1 trading days at 1H)

NO indicators (no RSI on price, no MACD, no Bollinger). Pure structural trade
on the VIX term structure curve. The CPCV-tuned thresholds are slightly tighter
than the defaults in apex/strategies/vix_term_structure.py.

DEPLOYMENT NOTES
----------------

1. **Paper-trade for 4 weeks minimum** before live capital — verify execution
   costs (slippage, commission) don't eat the edge.
2. **Start at 1/4 of intended size** — even with Sharpe 3+, real markets surprise.
3. **Kill switch**: pause if live drawdown exceeds 2× backtest MaxDD (so -6%).
4. **Re-tune quarterly** — VRP-curve dynamics drift; refresh params every 3 months
   on the latest tune window via `python apex.py --ensemble --budget heavy`.

KNOWN RISKS
-----------

- Concentrated bet on VIX term structure mechanics. If a regulatory or
  structural change disrupts how VIX/VIX3M interact (e.g., another XIV-style
  blowup, new vol product crowding), edge could die quickly.
- 108 holdout trades is statistically meaningful but not deep. A single
  large drawdown event could materially change the live Sharpe estimate.
- The cross_asset_vol_overlay multiplier scales positions but didn't
  contribute meaningfully in this validation (multiplier ≈ 1.0 most of
  the period).

USAGE
-----

Reproduce this exact validation:
    python apex.py --ensemble --budget heavy --config configs/production/vix_term_v1_config.json

Programmatic instantiation:
    from strategies.production.vix_term_structure_v1 import make_strategy
    strat = make_strategy()
    # strat is a VIXTermStructureStrategy with TUNED_PARAMS pre-applied
"""

from typing import Any, Dict

# Tuned parameters from Phase 13 heavy-budget CPCV search (2026-04-26)
TUNED_PARAMS: Dict[str, Any] = {
    "contango_extreme_threshold":     0.9176,
    "backwardation_extreme_threshold": 1.1408,
    "neutral_low":                     0.9463,
    "neutral_high":                    1.0004,
    "stop_atr_mult":                   1.6433,
    "max_hold_bars":                   8,
}

VALIDATION_METADATA: Dict[str, Any] = {
    "validated_at":      "2026-04-26T17:42:36",
    "pipeline_version":  "Phase 16",
    "tune_window_bars":  7530,
    "holdout_window_bars": 2510,
    "tune": {
        "layer_b_cpcv_median_sharpe": 5.00,
        "layer_b_cpcv_iqr":           [4.41, 5.60],
        "layer_b_cpcv_pct_positive":  1.00,
        "n_trades":                   345,
        "win_rate_pct":               58.0,
        "total_return_pct":           280.94,
        "max_dd_pct":                 -3.91,
        "sharpe_annualized":          4.26,
        "calmar":                     71.80,
    },
    "holdout_true_oos": {
        "layer_b_cpcv_median_sharpe": 3.85,
        "layer_b_cpcv_iqr":           [2.95, 4.57],
        "layer_b_cpcv_pct_positive":  1.00,
        "n_trades":                   108,
        "win_rate_pct":               58.4,
        "total_return_pct":           37.68,
        "max_dd_pct":                 -3.00,
        "sharpe_annualized":          3.27,
        "calmar":                     12.58,
    },
    "universe":         ["SPY", "QQQ"],
    "data_window":      ["2021-01-01", "2026-04-26"],
    "exec_timeframe":   "1H",
    "data_sources":     ["Polygon Stocks Starter", "FRED VIXCLS", "FRED VXVCLS"],
}


def make_strategy():
    """Return a VIXTermStructureStrategy instance with tuned params applied."""
    from apex.strategies.vix_term_structure import VIXTermStructureStrategy
    return VIXTermStructureStrategy(params=TUNED_PARAMS)


if __name__ == "__main__":
    import json
    print(__doc__.strip())
    print("\n=== TUNED_PARAMS ===")
    print(json.dumps(TUNED_PARAMS, indent=2))
    print("\n=== VALIDATION_METADATA ===")
    print(json.dumps(VALIDATION_METADATA, indent=2))
