# Changelog

## [3.0.0] — 2026-04 — Institutional Multi-Strategy Ensemble

### Added
- `apex/strategies/` package with `StrategyBase` ABC and `STRATEGY_REGISTRY`
- 6 institutional strategies registered via `@register_strategy` decorator:
  - `vrp_gex_fade` — Strategy 1: VRP + GEX gamma-wall fade in suppressed-vol regimes
  - `opex_gravity` — Strategy 2: max-pain magnetism during OPEX week
  - `vix_term_structure` — Strategy 3: VIX/VIX3M curve mean-reversion
  - `vol_skew_arb` — Strategy 4: 25-delta put/call IV skew arbitrage
  - `smc_structural` — Strategy 5: FVG + Order Block confluence (VIX/VPIN gated)
  - `cross_asset_vol_overlay` — Strategy 6: VIX/MOVE/OVX percentile size multiplier
- `apex/ensemble/` package: risk-parity weighting, regime overlay, combiner
  (max 30% per strategy, 60-day rolling vol, R1/R2/R3/R4 tilts, R4 -> 0)
- `apex/validation/ensemble_cpcv.py` — portfolio-level CPCV at 1H frequency
- `apex/validation/walk_forward.py` — dynamic vs static weight comparison
- `apex/data/options_chain.py`, `options_gex.py`, `opex_calendar.py`,
  `vol_skew.py`, `cross_asset_vol.py`, `dealer_levels.py`, `order_blocks.py`
- `apex/main_ensemble.py` — full ensemble pipeline orchestrator with
  `prepare_ensemble_data`, `run_layer_a/b/c_validation`
- `apex/report/ensemble_report.py` — 7-tab Plotly HTML report
- `--ensemble` CLI flag opt-in to the institutional pipeline
- `ensemble` block in `apex_config.json` (max_weight, vol_lookback_days,
  size_change_threshold, curated strategies list)
- Three-layer validation gate: Layer A per-strategy CPCV, Layer B portfolio
  CPCV (PASS = median Sharpe > 0.8 AND > 65% folds positive), Layer C
  walk-forward weights (PASS = uplift >= 0.05)

### Outputs (ensemble mode)
- `strategy_layer_a_results.csv`, `ensemble_layer_b_results.json`,
  `ensemble_layer_c_results.json`, `ensemble_report.html`

### Preserved
- Legacy single-strategy pipeline (no `--ensemble` flag) is byte-for-byte
  unchanged: golden snapshot intact, `python apex.py --test` still works,
  `--strategy <file>` mode untouched.
- `DEFAULT_ARCHITECTURE` and `DEFAULT_PARAMS` unchanged.
- All previous v2.0 tests pass; total test count grows from 252 to 276.

### Test counts
- v2.0: 130 tests
- v2.x (Phases 12A-12G): 252 tests
- v3.0 (this release): 276 tests

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
