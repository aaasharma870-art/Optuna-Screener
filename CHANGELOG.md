# Changelog

## [2.0.0] — 2026-04

### Added
- `apex/` package: modularized from 3215-line `apex.py`
- FRED client for VIX/VXV/VXN/GVZ (macro volatility data)
- VRP Regime classifier: R1 (suppressed fade), R2 (reduced fade), R3 (amplified trend), R4 (crisis)
- Long/short execution with symmetric stops + borrow-fee accrual
- VWAP sigma-bands, VPIN (volume-bucketed BVC), VWCLV, FVG detector, RSI2
- Dynamic FVG trailing stops with ATR fallback
- Cross-asset basket momentum position-size multiplier
- Multi-objective Pareto optimization (NSGA-II) with regime-specific fitness
- Synthetic price-path Monte Carlo (block bootstrap)
- CPCV, Deflated Sharpe Ratio, Probability of Backtest Overfitting
- `--validate-vrp` CLI smoke test
- `strategy_mode: "vrp_regime"` config dispatch
- Golden-snapshot regression harness + 130 tests
- `POLYGON_API_KEY` + `FRED_API_KEY` env-var overrides

### Preserved
- Optuna Layers 1 / 2 / 3
- 25% True Holdout split
- Look-ahead prevention (signal <= bar i, fill at open[i+1])
- Polygon retry + cache
- Checkpoint/resume
- All 14 classic indicators
- HTML report + AmiBroker AFL
