# Institutional Multi-Strategy Ensemble — Design Spec

**Date:** 2026-04-26
**Status:** Approved — ready for implementation plan
**Scope:** Build a 6-strategy structural-primitives ensemble on top of the existing
Optuna Screener pipeline. Combine via risk parity + regime overlay. Validate at
both per-strategy and portfolio level via CPCV.

---

## 1. Overview & Purpose

The current pipeline produces a single VRP-fade strategy with median CPCV Sharpe
0.38 on SPY (real but modest edge). The user wants institutional-tier returns
(Sharpe 1.5-2.5+). The path: stack 5-6 uncorrelated structural alpha sources
into an ensemble.

**Critical user direction:** strategies must be based on **institutional structural
primitives** (gamma exposure, OPEX mechanics, vol surface, term structure, smart-
money concepts), NOT retail technical indicators (RSI, Bollinger, MACD, etc.).
"If it were as easy as indicators everyone would do it." — User, 2026-04-26.

**Realistic outcome target:** ensemble CPCV median Sharpe **1.0-1.5** with the
current data tier (Polygon Stock Starter + Options Starter, FRED). Pushing past
this requires alternative data and/or higher-frequency execution (deferred to
later phases).

---

## 2. Goals & Non-Goals

**In scope:**
- 6 strategy modules under `apex/strategies/`, each implementing a `StrategyAdapter`-style interface
- Ensemble combiner under `apex/ensemble/` using risk parity + regime overlay
- Per-strategy CPCV validation (Layer A) and ensemble CPCV (Layer B)
- Walk-forward weight validation (Layer C)
- Single end-to-end pipeline command that produces ensemble HTML report

**Out of scope:**
- Alternative data integration (sentiment, dark pool, satellite) — Workstream D
- Higher-frequency execution (5-min or tick) — Workstream E
- Walk-forward continuous re-tuning of strategy params — Workstream B (deferred)
- ML regime classification — Workstream C (deferred)
- Crypto / FX / futures (out of equity-only scope per user choice)
- Real-time live trading deployment

---

## 3. Preserved Invariants

These existing pipeline guarantees MUST remain intact:

- Optuna Layers 1 / 2 / 3 (each strategy still uses Layer 2 for its own params)
- 25% True Holdout split happens BEFORE all optimization
- Look-ahead prevention (signal ≤ bar `i`, fill at `open[i+1]`)
- Polygon retry/cache + FRED cache
- Existing 14 indicators in `basics.py` (preserved verbatim)
- Existing VRP pipeline (`apex/regime/vrp_regime.py`, `apex/data/fred_client.py`,
  `apex/data/options_gex.py`) — strategy 1 builds on top of this
- Golden snapshot regression test must continue to pass
- All 173 existing tests must continue to pass

---

## 4. Architecture Overview

### 4.1 Target Package Layout

```
apex/
  strategies/                              NEW package
    __init__.py                            registry: STRATEGY_REGISTRY = {name: cls}
    base.py                                StrategyBase abstract class (entry_fn, exit_fn, get_params, get_data_requirements)
    vrp_gex_fade.py                        Strategy 1
    opex_gravity.py                        Strategy 2
    vix_term_structure.py                  Strategy 3
    vol_skew_arb.py                        Strategy 4
    smc_structural.py                      Strategy 5
    cross_asset_vol_overlay.py             Strategy 6 (overlay; multiplies sizing)

  ensemble/                                NEW package
    __init__.py
    risk_parity.py                         compute_risk_parity_weights(strategies, lookback_days=60)
    regime_overlay.py                      apply_regime_tilts(weights, current_regime)
    combiner.py                            EnsembleCombiner.run() — orchestrates per-strategy
                                           backtest + risk-parity + overlay + final positions

  validation/                              EXTEND existing
    ensemble_cpcv.py                       NEW: CPCV at the portfolio level

  main.py                                  EXTEND: --ensemble flag runs the new pipeline
                                           instead of the legacy single-strategy mode

tests/
  test_strategy_base.py                    base interface tests
  test_strategy_vrp_gex.py                 strategy 1
  test_strategy_opex.py                    strategy 2
  test_strategy_vix_term.py                strategy 3
  test_strategy_vol_skew.py                strategy 4
  test_strategy_smc.py                     strategy 5
  test_strategy_overlay.py                 strategy 6
  test_ensemble_risk_parity.py             weight computation
  test_ensemble_regime_overlay.py          tilt logic
  test_ensemble_combiner.py                full pipeline integration
  test_ensemble_cpcv.py                    portfolio-level CPCV
```

### 4.2 Strategy Interface Contract

Each strategy module defines:

```python
class Strategy(StrategyBase):
    name = "vrp_gex_fade"
    data_requirements = ["exec_df_1H", "options_chain_daily", "vix", "vix3m"]

    def compute_signals(self, data: dict) -> pd.DataFrame:
        """Return DataFrame with columns: entry_long, entry_short, exit_long, exit_short.
        Each is a bool/int Series indexed like the exec_df.
        """

    def compute_position_size(self, data: dict, signals: pd.DataFrame) -> pd.Series:
        """Return per-bar position size (-1.0 to +1.0). Risk-parity will scale this."""

    def get_tunable_params(self) -> dict:
        """Optuna search space for this strategy. {param_name: (lo, hi, type)}"""
```

Each strategy is independently optimizable via Optuna (Layer 2 runs per strategy).

---

## 5. The 6 Strategies — Concrete Specifications

### 5.1 Strategy 1: VRP + GEX Fade (enhanced)

**Structural primitive:** Real options-derived gamma walls (Call Wall / Put Wall)
combined with VRP percentile filter.

**Data requirements:**
- 1H exec bars (existing Polygon)
- Daily options chain → Call Wall, Put Wall, Gamma Flip (existing `compute_gex_proxy`)
- VIX, VIX3M, VRP percentile (existing `compute_vrp`)

**Entry rules:**
- VRP percentile > 70 (suppressed regime, doc spec)
- VIX/VIX3M < 0.95 (contango)
- VIX absolute < 25
- LONG: price within 0.5σ above Put Wall AND RSI(2) < 15 AND VPIN percentile < 50
- SHORT: price within 0.5σ below Call Wall AND RSI(2) > 85 AND VPIN percentile < 50

**Exit rules:**
- Touch session VWAP (primary target)
- Opposite gamma wall reached (alternate target)
- Stop: 1.0 ATR beyond entry
- Time stop: 21 bars (~3 trading days at 1H)

**Tunable params (Optuna Layer 2):**
- `vrp_pct_threshold`: int(60, 90)
- `gamma_wall_proximity_atr`: float(0.2, 1.0)
- `rsi2_oversold`: int(5, 25)
- `rsi2_overbought`: int(75, 95)
- `vpin_pct_max`: int(40, 60)
- `stop_atr_mult`: float(0.6, 1.6)
- `max_hold_bars`: int(7, 35)

**Expected Sharpe target (individual):** 0.5-0.9

### 5.2 Strategy 2: OPEX Gravity

**Structural primitive:** Max-pain magnetism + post-OPEX vol release. Predictable
gamma-induced pinning around monthly options expiration (3rd Friday).

**Data requirements:**
- 1H exec bars
- Options chain → identify high-OI strike near current price (max-pain proxy)
- Calendar awareness (week-of-OPEX flag)

**Entry rules (OPEX-week strategy, Tue/Wed entry):**
- Current trading week contains 3rd Friday (OPEX week)
- Day-of-week ∈ {Tuesday, Wednesday}
- Identify `pin_strike` = the strike with highest combined call+put OI within ±5% of current spot, using the next-Friday-expiry options chain
- LONG if `(pin_strike - spot) / spot > 0.005` (price below pin, expect upward gravity)
- SHORT if `(spot - pin_strike) / spot > 0.005` (price above pin)

**Exit rules:**
- Pin strike touched within ±0.2%
- Friday close (forced exit before weekend)
- Stop: 1.5% beyond entry

**Tunable params:**
- `min_pin_distance_pct`: float(0.003, 0.015)
- `pin_strike_window_pct`: float(0.03, 0.08) — window around spot to scan for highest-OI strike
- `entry_dow`: categorical(["Mon-Tue", "Tue-Wed", "Wed-Thu"])
- `forced_exit_dow`: categorical(["Thu", "Fri-mid", "Fri-close"])

**Expected Sharpe target:** 0.4-0.8 (low-frequency, ~12 trades/year per symbol)

### 5.3 Strategy 3: VIX Term Structure Trade

**Structural primitive:** VIX/VIX3M ratio mean-reversion. Trades the curve, not
the level.

**Data requirements:**
- VIX, VIX3M daily (FRED, existing)
- 1H exec bars on SPY (instrument)

**Entry rules:**
- Compute `ts_ratio = VIX / VIX3M`, daily
- LONG SPY when `ts_ratio` enters extreme contango (`< 0.85`) — vol curve overpriced near term, expect mean-reversion
- SHORT SPY when `ts_ratio` enters extreme backwardation (`> 1.10`) — fear spike, expect reversal
- Confirmation: 5-day RSI on `ts_ratio` itself (oversold/overbought)

**Exit rules:**
- `ts_ratio` reverts to neutral band (0.95-1.02)
- Time stop: 10 bars (~1.5 trading days at 1H)
- Stop: 1.5 ATR beyond entry

**Tunable params:**
- `contango_extreme_threshold`: float(0.80, 0.92)
- `backwardation_extreme_threshold`: float(1.05, 1.20)
- `neutral_low`: float(0.93, 0.97)
- `neutral_high`: float(1.00, 1.04)
- `stop_atr_mult`: float(0.8, 2.0)
- `max_hold_bars`: int(5, 30)

**Expected Sharpe:** 0.4-0.8 (low-frequency, structural)

### 5.4 Strategy 4: Volatility Skew Arbitrage

**Structural primitive:** 25-delta put IV / 25-delta call IV ratio extremes.
Trades asymmetric fear pricing in the options surface.

**Data requirements:**
- Polygon options chain (Options Starter has IV per contract)
- 1H exec bars on underlying (SPY)
- New helper: `compute_skew_ratio(chain, dte=30)` extracts 25-delta P/C IV

**Entry rules:**
- Compute `skew = IV_25dPut / IV_25dCall` daily for 30-DTE options
- LONG SPY when `skew > 1.30` (extreme put fear, mean-reversion long)
- SHORT SPY when `skew < 0.95` (extreme call greed, mean-reversion short)

**Exit rules:**
- Skew reverts to normal band (1.05-1.20)
- Time stop: 5 trading days
- Stop: 1.0 ATR beyond entry

**Tunable params:**
- `put_skew_extreme`: float(1.20, 1.50)
- `call_skew_extreme`: float(0.85, 1.00)
- `normal_low`: float(1.00, 1.10)
- `normal_high`: float(1.15, 1.30)
- `dte_target`: int(20, 45)
- `stop_atr_mult`: float(0.7, 1.5)
- `max_hold_days`: int(2, 10)

**Expected Sharpe:** 0.6-1.0 (the skew dislocation trade is genuinely structural)

### 5.5 Strategy 5: SMC Structural (FVG + Order Block)

**Structural primitive:** Pure price-structure entries. No indicators. Uses
unfilled FVGs and order blocks as confluence zones.

**Data requirements:**
- 1H exec bars (existing FVG detector from Phase 2)
- VIX (filter for low-vol confluence)
- VPIN percentile (filter for noise environment, smart money latent)

**Entry rules:**
- LONG: price retests an unfilled bullish FVG OR bullish order block
       (defined as: 3-bar pattern where bar[i] = down close, bar[i+1] = inside or
       small body, bar[i+2] = strong up close > bar[i].open)
       AND VIX < 25 AND VPIN percentile < 50
       AND price is within FVG zone (between FVG low and high)
- SHORT: mirror for bearish FVG / order block

**Exit rules:**
- FVG fully filled (close below FVG low for bullish, above FVG high for bearish)
- Opposite-direction FVG forms
- Time stop: 16 bars (~2 trading days at 1H)
- Dynamic stop: nearest unfilled FVG behind price (Phase 6 stops module)

**Tunable params:**
- `vix_filter_max`: int(15, 30)
- `vpin_pct_max`: int(30, 60)
- `ob_min_body_ratio`: float(0.3, 0.8)
- `max_hold_bars`: int(8, 24)
- `dynamic_stop`: categorical([True, False])

**Expected Sharpe:** 0.4-0.8

### 5.6 Strategy 6: Cross-Asset Vol Regime Overlay

**Not a standalone strategy** — multiplies position sizes of strategies 1-5.

**Structural primitive:** Macro vol regime context across asset classes.

**Data requirements:**
- VIX (equity vol, FRED)
- MOVE (rates vol, FRED `BAMLH0A0HYM2EY` or proxy)
- OVX (oil vol, FRED `OVXCLS`)

**Logic:**
```
For each bar:
  vol_inputs = [VIX, MOVE, OVX]  (each normalized to its 252-day percentile)
  if all three percentiles > 80:  size_mult = 0.5   # risk-off, all assets stressed
  elif all three percentiles < 20:  size_mult = 1.2   # risk-on, low vol everywhere
  elif divergent (VIX high, MOVE low):  size_mult = 1.0   # equity-specific stress, normal sizing
  else: size_mult = 1.0
```

Applied as `final_position = strategy_position * size_mult`. No tunable params (hardcoded thresholds; could add later).

**Expected impact:** +0.1-0.3 Sharpe from regime-aware sizing.

---

## 6. Ensemble Combiner Logic

### 6.1 Per-Bar Algorithm

```
1. Each strategy s in {1..5} computes its signal s.compute_signals(data) and
   raw position s.compute_position_size(data, signals).
2. Compute strategy 6's regime_size_mult (scalar).
3. For each strategy s, compute rolling 60-day annualized vol of its returns:
       vol_s = sqrt(252) * std(strategy_s_returns_last_60d)
4. Risk-parity weights (rebalanced monthly, not per-bar):
       w_s = (1/vol_s) / sum(1/vol_j for j in 1..5)
5. Apply regime tilts based on current regime_state:
       if regime in {Contango_Calm, R1, R2}:    w_s *= 1.2 if s in {1, 4, 5}
       if regime in {Backwardation, R3}:         w_s *= 1.2 if s in {2, 3}
       Renormalize so sum(w_s) == 1.0
6. Cap any w_s at 0.30 (prevent single strategy dominance)
7. Final position per symbol:
       portfolio_pos[symbol] = sum(w_s * raw_position_s[symbol]) * regime_size_mult
8. Fire trade if portfolio_pos changes by > 0.10 (10% delta threshold to avoid churn)
```

### 6.2 Risk Parity Weight Computation

`apex/ensemble/risk_parity.py::compute_risk_parity_weights`:
- Inputs: dict[strategy_name → equity_curve_returns] over rolling 60-day window
- Outputs: dict[strategy_name → weight in [0, 0.30]] summing to 1.0
- Edge case: strategy with zero or NaN vol gets weight 0; remaining renormalize.

### 6.3 Regime Overlay

`apex/ensemble/regime_overlay.py::apply_regime_tilts`:
- Inputs: weights dict, current_regime string (one of R1/R2/R3/R4 from existing classifier)
- Tilts:
  - R1, R2 (suppressed/calm): boost mean-reversion strategies (1, 4, 5) by 20%
  - R3 (amplified): boost trend/term-structure strategies (2, 3) by 20%
  - R4 (no-trade): all weights → 0 (sit out)
- Renormalize after tilts.

### 6.4 Capacity Guardrails

- Max 30% of portfolio in any single strategy (hard cap).
- Max 100% combined exposure (no leverage in v1).
- Min 5% of portfolio in any positive-Sharpe strategy (avoid concentration).

---

## 7. Validation Pipeline

### 7.1 Layer A — Per-Strategy CPCV

Each strategy must individually pass:
- Median CPCV OOS Sharpe > 0.3 across 28 folds
- > 55% of folds with positive Sharpe
- DSR > 0.6
- At least 50 trades total in the holdout window

If a strategy fails Layer A → it is **excluded from the ensemble** (do not deploy it).

### 7.2 Layer B — Ensemble CPCV

After all surviving strategies are combined via risk parity + regime overlay:
- Run 28-fold CPCV on the COMBINED portfolio NAV curve
- Target: median Sharpe > 0.8, > 65% folds positive
- Report ensemble DSR, max DD distribution, return distribution

### 7.3 Layer C — Walk-Forward Weights Validation

The risk-parity weights are recomputed monthly. We validate that this dynamic
weighting actually adds value vs static (initial) weights:
- For each month M in the backtest:
  - Compute weights from data ≤ M-1 month
  - Apply to month M's per-strategy returns
  - Compare cumulative performance vs static weights from month 0
- Pass if dynamic weights produce ≥ 0.05 Sharpe uplift over static.

---

## 8. Implementation Order

Sequential, each gated on prior:

| # | Step | Effort (eng-days) | Gate |
|---|---|---|---|
| 1 | Build `apex/strategies/base.py` interface + `apex/ensemble/` framework | 3 | Tests for interface contract |
| 2 | Strategy 1 (VRP+GEX fade enhanced from existing) | 2 | Layer A pass |
| 3 | Strategy 2 (OPEX gravity) | 3 | Layer A pass |
| 4 | Strategy 3 (VIX term structure) | 2 | Layer A pass |
| 5 | Strategy 4 (Vol skew arb) — most complex options work | 4 | Layer A pass |
| 6 | Strategy 5 (SMC structural extension) | 3 | Layer A pass |
| 7 | Strategy 6 (Cross-asset vol overlay) | 1 | Logic test |
| 8 | Ensemble integration + Layers B/C validation | 4 | Layer B + C pass |
| 9 | HTML report + CLI integration | 2 | End-to-end run |

**Total: ~24 working days = 5 weeks for fully validated production ensemble.**

---

## 9. Realistic Sharpe Target

If each strategy passes Layer A with individual Sharpe 0.5-0.8 (realistic for
structural strategies) and they exhibit 30-50% pairwise correlation:

- Per portfolio math: ensemble Sharpe ≈ avg_individual_Sharpe × √(N / (1 + (N-1)·avg_corr))
- 5 strategies × 0.6 avg Sharpe × correlation 0.4 → ensemble Sharpe ≈ **1.05**
- 5 strategies × 0.7 avg Sharpe × correlation 0.3 → ensemble Sharpe ≈ **1.40**

**Realistic ensemble CPCV median Sharpe: 1.0-1.5.**

This is a **2.5-4x improvement** over the current 0.38 baseline. Reaches "good
systematic strategy" tier. Beyond this requires alt data (Workstream D),
higher-frequency execution (Workstream E), or ML regime classification
(Workstream C) — all explicitly deferred.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Strategies more correlated than expected → Sharpe lift smaller | Layer C validates correlation assumptions; will surface if real correlation > 0.6 |
| Optuna over-tunes individual strategies → Layer A passes but Layer B fails | CPCV rigorously catches per-strategy overfitting; only Layer A survivors enter ensemble |
| Regime overlay misclassifies → reduces Sharpe instead of boosting | Validate regime overlay's marginal contribution separately; can disable if it hurts |
| Vol skew strategy needs more options-chain history than Polygon Options Starter provides (2y) | Document data limitation; vol skew may be excluded from ensemble if insufficient holdout |
| OPEX gravity is well-known retail trade → real edge gone | Validate on out-of-sample 2020-2022 vs 2023-2025; if edge halved → retire strategy |
| Ensemble combiner has subtle look-ahead via vol estimation | Strict shift(1) on rolling vol; explicit unit test checks for look-ahead |
| Implementation takes longer than 5 weeks | Each strategy is independent; ship strategies 1-3 first as a 3-strategy ensemble (~2 weeks) |

---

## 11. Out of Scope (explicit)

- Alternative data (sentiment, news, dark pool)
- Higher-frequency execution (5-min, tick)
- ML regime classification
- Walk-forward continuous param re-tuning of strategies (params fixed after Layer A passes)
- Crypto / FX / futures
- Real-money live trading
- AmiBroker integration for ensemble (legacy AmiBroker stays single-strategy)
- Live broker integration

---

## 12. Decisions Log

| Decision | Rationale |
|----------|-----------|
| Equity-only scope (SPY, QQQ, GLD) | User chose tight scope; lower complexity |
| Structural primitives only, no indicators | User direction: "if it were as easy as indicators everyone would do it" |
| Risk parity + regime overlay | Standard institutional approach; robust to estimation noise |
| 30% per-strategy cap | Prevent single-strategy dominance; institutional norm |
| Layer A threshold: median Sharpe > 0.3 | Low enough to admit strategies that improve ensemble even if marginal individually |
| Layer B threshold: ensemble Sharpe > 0.8 | "Beat-the-market" tier — minimum justifiable for deployment |
| Sequential implementation order | Allows early validation of each strategy independently |
| 60-day rolling window for risk-parity vol | Long enough to be stable, short enough to adapt to regime change |
| Monthly weight rebalance | Avoids per-bar churn; matches institutional rebalancing cadence |
| 6th strategy as overlay (not standalone) | Vol regime context is naturally a sizer, not a signal |
| CPCV at both per-strategy and ensemble level | Each layer catches different overfitting modes |

---

## 13. Open Items (zero at time of approval)

All prerequisite decisions resolved. Implementation plan is the next document.
