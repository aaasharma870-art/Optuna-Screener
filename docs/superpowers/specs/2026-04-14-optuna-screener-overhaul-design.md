# Optuna Screener — Institutional Arbitrage Engine Overhaul

**Date:** 2026-04-14
**Status:** Approved — ready for implementation plan
**Scope:** Full architectural overhaul of `apex.py` into a modular, multi-directional,
regime-aware, statistically-hardened research pipeline. Spot equities only (long and
short). No options execution.

---

## 1. Overview & Purpose

The current `apex.py` is a 3215-line single-file Optuna-driven research pipeline. It is
statistically rigorous (Optuna 3-layer + 25% True Holdout) but is limited to long-only
spot equity trading with a basic indicator set and a single EMA/ATR-based regime gate.

This overhaul upgrades it into an institutional-grade engine with:
- Six-quadrant volatility-regime matrix (VIX-term-structure × Variance-Risk-Premium)
- Long/Short directional execution with borrow-fee-aware short PnL
- Advanced indicators (VWAP σ-bands, VPIN, VWCLV, FVG)
- FVG-anchored dynamic trailing stops
- Cross-asset basket momentum with position-size scaling
- Multi-objective Pareto optimization with regime-specific fitness
- Four additional statistical-robustness validators (Synthetic MC, CPCV, DSR, PBO)

The entire overhaul preserves every existing validation layer. No statistical guardrail
is removed; only added.

---

## 2. Goals & Non-Goals

**In scope:**
- Long and short equity trading, spot only
- Modularization of `apex.py` into an `apex/` package
- New data ingestion for macro vol (yfinance) and options-derived GEX (Polygon)
- Full regime-matrix rewrite
- Four new indicators (registered, opt-in via architecture choices)
- Execution engine rewrite for direction-awareness and FVG trailing stops
- Layer 2 multi-objective optimization
- Regime-specific fitness functions
- Synthetic-price-path MC, CPCV, DSR, PBO
- Golden-snapshot regression tests to protect the legacy long-only code path

**Out of scope:**
- Options execution (Iron Condor simulation, strike selection, IV surface simulation)
- SpotGamma subscription integration (replaced by computing our own GEX proxy)
- Futures data (ES, NQ, GC, CL, ZN) — replaced with ETF proxies (SPY, QQQ, GLD, USO, IEF)
- n8n orchestration / Walk-Forward Matrix (separate effort; out of apex.py repo)
- Live trading integration beyond AmiBroker AFL export

---

## 3. Preserved Invariants (non-negotiable)

The following must pass byte-equality regression tests before any phase can be considered
complete:

- Optuna Layer 1 architecture search (TPE multivariate)
- Optuna Layer 2 deep parameter tune
- Optuna Layer 3 robustness gauntlet (MC, noise, regime-stress, param-sensitivity)
- 25% True Holdout split; split happens before Layer 1 runs
- Every downstream function keeps `(exec_df, exec_df_holdout)` separation
- Look-ahead prevention: signal timestamp ≤ bar `i`, fill at `open[i+1]`
- Polygon retry/back-off + on-disk caching
- Checkpoint system per stage + `--resume` flag
- All existing classic indicators remain available in `basics.py` (EMA, ATR, VWAP, RSI, MACD, BB, Stoch, OBV, ADX, CCI, Williams R, Keltner, VolSurge, Parkinson IV proxy — 14 total preserved verbatim)
- HTML report tabs + AmiBroker AFL export preserved in shape

---

## 4. Target Architecture

### 4.1 Package Layout

```
apex/
  __init__.py
  main.py                   CLI entry, orchestration (replaces apex.py main())
  config.py                 load_config + env-var override (POLYGON_API_KEY wins)
  logging_util.py           log() + eta_str()

  data/
    polygon_client.py       polygon_request, fetch_daily, fetch_bars
    macro_vol.py            fetch_macro_volatility — VIX, VIX3M via yfinance
    options_gex.py          Call Wall / Put Wall / Gamma Flip / Vol Trigger /
                            Abs Gamma Strike from Polygon options OI + greeks
    dealer_levels.py        ingest_flux_points(df) — merge GEX onto exec df
    cross_asset.py          fetch basket (SPY, QQQ, GLD, USO, IEF)

  indicators/
    basics.py               EMA, ATR, VWAP (legacy single line), RSI, MACD, BB,
                            Stoch, OBV, ADX, CCI, Williams R, Keltner, VolSurge,
                            Parkinson IV proxy (preserved verbatim)
    vwap_bands.py           VWAP + 1σ/2σ/3σ volume-weighted variance bands
    vpin.py                 Bulk Volume Classification VPIN
    vwclv.py                Volume-Weighted Close Location Value + 5-bar cumulative
    fvg.py                  3-bar imbalance detector + fill-tracking

  regime/
    realized_vol.py         20-day realized vol (log-return based)
    vrp.py                  Variance Risk Premium = IV30 − realized_vol_20d
    six_quadrant.py         6-quadrant regime classifier + daily→1H merge

  engine/
    backtest.py             run_backtest (direction-aware, FVG-stop-aware)
    portfolio.py            Cross-asset basket + size multiplier logic
    fees.py                 Borrow-fee model for shorts (bps/day accrual)
    stops.py                Dynamic FVG trailing-stop logic (+ ATR fallback)

  optimize/
    layer1.py               Architecture search (indicator / exit / regime choice)
    layer2.py               Deep tune with multi-objective Pareto
    layer3.py               Robustness gauntlet (preserved + synthetic MC hook)
    fitness.py              Regime-specific fitness functions

  validation/
    synthetic_mc.py         Block-bootstrap synthetic price-path MC
    cpcv.py                 Combinatorial Purged Cross-Validation
    dsr.py                  Deflated Sharpe Ratio
    pbo.py                  Probability of Backtest Overfitting

  report/
    html_report.py          Plotly HTML report generation
    csv_json.py             trades.csv, summary.csv, parameters.json
    amibroker.py            AFL generation + optional COM push

  util/
    checkpoints.py          save_checkpoint / load_checkpoint
    concept_parser.py       parse_concept (preserved)
    sector_map.py           SECTOR_MAP constant

apex.py                     Thin shim: `from apex.main import main; main()`

tests/
  conftest.py
  fixtures/
    SPY_1H.parquet          frozen 180-bar slice
    QQQ_1H.parquet          frozen 180-bar slice
    SPY_daily.parquet       frozen daily history for regime tests
    options_chain_sample.json
  test_regression_golden.py Legacy-path byte-equality
  test_backtest_math.py     Long/Short PnL, borrow fee, FVG stop
  test_indicators.py        Determinism on fixtures
  test_regime.py            Boundary values (0.95, 1.02, percentile 30/70/20/80)
  test_fitness.py           Formula correctness
  test_synthetic_mc.py      Block bootstrap statistical sanity
  test_cpcv.py              No-leakage between folds
  test_dsr.py               Formula reproduction vs published example
  test_pbo.py               Known-overfit and known-robust corner cases
```

### 4.2 Module Boundary Rules

- `indicators/` modules are pure functions. No config access. Input: dataframe + params.
  Output: Series or DataFrame.
- `regime/` consumes macro-vol and realized-vol; emits regime columns. Pure given inputs.
- `engine/backtest.py` consumes `(df, signals, architecture, params)` and returns
  `(trades, stats)`. It does not know about Optuna.
- `optimize/` modules orchestrate Optuna and call `engine.backtest`. No direct data
  fetching inside Optuna trials.
- `validation/` modules consume trade sequences and/or price series. No Optuna coupling.
- `data/` modules do all network I/O and all caching. Nothing else writes to
  `apex_cache/`.

---

## 5. Phase 0 — Foundation

### 5.1 Phase 0a — Regression Test Harness

**Goal:** freeze the current pipeline's output so any future change that breaks the
legacy long-only code path is caught immediately.

**Deliverables:**
1. `tests/fixtures/SPY_1H.parquet` and `QQQ_1H.parquet` — last 180 bars of current
   Polygon cache (or fetched once and frozen). Committed.
2. `tests/fixtures/SPY_daily.parquet` and `QQQ_daily.parquet` for quick-screen.
3. `tests/conftest.py` with a `legacy_config` fixture that:
   - Uses mock Polygon data served from fixtures
   - Sets `arch_trials=3`, `deep_trials=5`, `mc_sims=100` (tiny budget)
   - Fixed `optuna.samplers.TPESampler(seed=42)`
   - Fixed `numpy.random.seed(42)`
4. `tests/test_regression_golden.py`:
   - Runs the full pipeline under `legacy_config` against the fixtures
   - Snapshots `trades_df`, `portfolio_stats`, `holdout_stats` to
     `tests/fixtures/golden/*.json`
   - On subsequent runs, asserts equality (tolerance 1e-9 for floats)
5. Config loader updated to accept env-var override:
   ```python
   api_key = os.environ.get("POLYGON_API_KEY") or cfg["polygon_api_key"]
   ```
6. `pytest`, `pytest-xdist`, and `pyarrow` (already present) in `requirements.txt`.

**Fixture freezing procedure:** run the current pipeline once with `POLYGON_API_KEY`
set, write outputs to `tests/fixtures/golden/`, commit. Future runs compare.

**Acceptance:** `pytest tests/test_regression_golden.py` passes on a fresh checkout.

### 5.2 Phase 0b — Modularization

**Goal:** move existing code into the package layout above without changing behavior.

**Procedure:**
1. Create `apex/` package structure.
2. Move each section of the old `apex.py` into its corresponding module.
3. Fix imports.
4. Leave top-level `apex.py` as:
   ```python
   from apex.main import main
   if __name__ == "__main__":
       main()
   ```
5. Re-run `pytest tests/test_regression_golden.py` — must still pass byte-equal.

**Acceptance:** golden test green, `python apex.py --test` produces identical output
to pre-move commit.

---

## 6. Phase 1 — Data Ingestion & Regime Matrix

### 6.1 `data/macro_vol.py`

```python
def fetch_macro_volatility(start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    """
    Returns a daily DataFrame indexed by date with columns:
      vix, vix3m, vix_ratio (vix/vix3m), realized_vol_20d, iv30, vrp
    Source: yfinance for ^VIX, ^VIX3M. IV30 derived from Polygon options chains
    (fallback to Parkinson estimator if Polygon call fails).
    Cached to {cache_dir}/macro_vol_{start}_{end}.parquet
    """
```

- Fetches `^VIX`, `^VIX3M` daily closes via yfinance
- IV30: pulls ATM 30-day-expiry call + put implied vols from Polygon,
  volume-weighted-averages to single IV30 value per day. Falls back to Parkinson
  if Polygon returns empty (e.g. weekend, holiday)
- Computes realized vol: `log_returns.rolling(20).std() * sqrt(252)`
- Computes VRP = IV30 − realized_vol_20d
- No `.shift()` here; regime module applies the shift

### 6.2 `data/options_gex.py`

```python
def compute_gex_proxy(symbol: str, as_of: date, cache_dir: Path) -> dict:
    """
    Returns:
      {
        'call_wall': float (strike with highest call gamma exposure),
        'put_wall': float (strike with highest put gamma exposure),
        'gamma_flip': float (strike where net gamma exposure crosses zero),
        'vol_trigger': float (0.85 × gamma_flip, empirical),
        'abs_gamma_strike': float (strike with highest |net gamma|),
      }
    """
```

- Fetches full options chain (all expiries within 45 DTE) from Polygon for `as_of`
- Computes per-strike per-contract GEX = OI × contract_size × spot² × gamma × 0.01
- Aggregates calls and puts separately; calls contribute +GEX, puts contribute −GEX
- Call Wall = argmax of call-side GEX magnitude
- Put Wall = argmax of put-side GEX magnitude
- Gamma Flip = interpolated zero-crossing of net cumulative GEX across strikes
- Cached per (symbol, date) in `{cache_dir}/gex/{symbol}_{date}.json`

### 6.3 `data/dealer_levels.py`

```python
def ingest_flux_points(exec_df: pd.DataFrame, symbol: str,
                      cache_dir: Path) -> pd.DataFrame:
    """
    Merges daily GEX levels onto the 1H exec df using .shift(1) as-of semantics.
    Adds columns: call_wall, put_wall, gamma_flip, vol_trigger, abs_gamma_strike
    """
```

- For each trading day in `exec_df`, looks up `compute_gex_proxy(symbol, prev_day)`
- Forward-fills intraday bars with previous-day levels (no look-ahead)

### 6.4 `regime/six_quadrant.py`

```python
def compute_regime(exec_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns to exec_df:
      regime_primary:  'Contango' | 'Neutral' | 'Backwardation'
      regime_secondary: 'Calm' | 'Elevated'
      regime_state:    f'{primary}_{secondary}'  (6 values)
    """
```

- Primary gate on shifted `vix_ratio`:
  - `< 0.95` → Contango, `0.95 ≤ x ≤ 1.02` → Neutral, `> 1.02` → Backwardation
- Secondary gate on VRP percentile (252-bar rolling, excluding current day):
  - `percentile ∈ [30, 70]` → Calm
  - `percentile < 20 or > 80` → Elevated
  - percentile ∈ `[20,30) ∪ (70,80]` → Calm (conservative; inside the "normal band")
- Regime columns are shift-1 before merge to prevent look-ahead
- Unit tests: exact boundary values (0.95, 1.02, pct 20/30/70/80), NaN propagation
  for days with insufficient history

**Replaces** the old EMA/ATR `compute_regime` entirely. Old regime param set
is removed from Optuna search space in Layer 1.

---

## 7. Phase 2 — Indicator Registry Upgrade

All indicators emit column(s) on the dataframe. All are pure functions. All are
registered in `INDICATOR_REGISTRY` so Layer 1 can select them.

### 7.1 `indicators/vwap_bands.py`

```python
def compute_vwap_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with columns: vwap, vwap_1σ_upper, vwap_1σ_lower,
    vwap_2σ_upper, vwap_2σ_lower, vwap_3σ_upper, vwap_3σ_lower
    """
```

- Reset cumulative sums at start of each session (daily boundary)
- Volume-weighted variance: `σ² = Σ(v_i · (tp_i − vwap)²) / Σv_i` where
  `tp = typical_price = (h+l+c)/3`

### 7.2 `indicators/vpin.py`

```python
def compute_vpin(df: pd.DataFrame, bucket_size: int = 50,
                window: int = 50) -> pd.Series:
    """
    Bulk Volume Classification VPIN. Returns percentile-ranked VPIN (0-100),
    where >60 = informed flow active, <50 = noise-driven.
    """
```

- Bulk Volume Classification: for each bar, `buy_vol = v · Z((c − c_prev) / σ_r)`
  where `Z` is the standard normal CDF and `σ_r` is the stdev of returns
- Sell volume = `v − buy_vol`
- VPIN = `|buy_vol − sell_vol|.rolling(window).sum() / v.rolling(window).sum()`
- Rolling 252-bar percentile rank emitted as the final column
- Guard: percentile is computed strictly from prior bars (no look-ahead)

### 7.3 `indicators/vwclv.py`

```python
def compute_vwclv(df: pd.DataFrame, ma_period: int = 20) -> pd.DataFrame:
    """
    Returns columns: vwclv (per-bar), vwclv_cum5 (5-bar cumulative CVD proxy).
    """
```

- Per bar: `clv = (close − low) / (high − low)`; range-0 bars → 0.5
- Weight: `w = volume / volume.rolling(ma_period).mean()`
- `vwclv = (2 · clv − 1) · w` (range −w to +w, negative = distribution, positive
  = accumulation)
- `vwclv_cum5 = vwclv.rolling(5).sum()`

### 7.4 `indicators/fvg.py`

```python
def detect_fvgs(df: pd.DataFrame) -> list[dict]:
    """
    Returns list of FVG records:
      {
        'start_idx': int (bar index where the 3-bar pattern starts),
        'end_idx': int (bar index of the middle bar),
        'direction': 'bullish' | 'bearish',
        'low': float (lower edge of gap),
        'high': float (upper edge of gap),
        'filled_at_idx': Optional[int] (None while unfilled, set on fill),
      }
    """
```

- Bullish FVG: `high[i] < low[i+2]` — gap between bar[i].high and bar[i+2].low
- Bearish FVG: `low[i] > high[i+2]` — mirror
- Fill detection: bullish FVG is filled the first time close falls back to
  `high[i]` or below; bearish FVG is filled when close rises to `low[i]` or above
- FVG list is returned in chronological order
- CRITICAL: FVG detection never produces entry signals. It only feeds `stops.py`.

---

## 8. Phase 3 — Execution Engine

### 8.1 Phase 3a — Long/Short Direction + Borrow Fees

`engine/backtest.py::run_backtest` gains `direction: Literal["long","short","neutral"]`
as part of `architecture`.

**Entry rules** (composed by Layer 1 from selected indicators):
- Long entry: `entry_score ≥ min_score_long` AND regime permits long
- Short entry: `entry_score ≤ −min_score_short` AND regime permits short
- `neutral` direction means both are allowed within the same trial

**PnL math:**
- Long: `pnl = exit_price − entry_price` per share
- Short: `pnl = entry_price − exit_price` per share (symmetric)
- Borrow fee on shorts: `fee_per_day = entry_price × annual_borrow_rate / 252`
  where `annual_borrow_rate` defaults to 0.02 (2%), configurable per symbol
  in `apex_config.json` under `borrow_rates: {symbol: rate}`
- Accrued fee is subtracted from short PnL at exit time, scaled by bars held / bars-per-day

**Stop losses (symmetric):**
- Long: `stop_price = entry × (1 − stop_pct)`
- Short: `stop_price = entry × (1 + stop_pct)`

**Regime-gated direction:**
- `Contango_Calm` / `Neutral_Calm` → favor `short` or `neutral` (fade setups)
- `Backwardation_*` / `*_Elevated` → favor `long` (trend / momentum setups)
- Gates applied through `engine/portfolio.py::permitted_directions(regime)`

**Config additions:**
```json
"borrow_rates": {
  "default": 0.02,
  "TSLA": 0.05,
  "GME": 0.30
}
```
Lookup order: `borrow_rates[symbol]` then `borrow_rates["default"]`.

### 8.2 Phase 3c — Dynamic FVG Trailing Stops

`engine/stops.py::compute_dynamic_stop(direction, price, unfilled_fvgs, atr,
atr_mult)`:
- Long position: returns the lower edge of the nearest un-filled bullish FVG below
  `price`, minus a small buffer (e.g. 0.05 × ATR). Falls back to `price − atr_mult × atr`
  if no FVG exists below.
- Short position: mirror — upper edge of nearest un-filled bearish FVG above, plus buffer.
- Fallback: ATR-multiplied distance when no FVG is present.

**Toggle:** `params["dynamic_stop"]` bool. When False, legacy static-percentage stop
is used (preserves legacy code path for regression test).

---

## 9. Phase 4 — Portfolio & Optimization

### 9.1 Phase 4a — Cross-Asset Basket Momentum

`engine/portfolio.py::compute_basket_alignment(basket_df, as_of_date) -> float`:
- Basket symbols (configurable): default `["SPY","QQQ","GLD","USO","IEF"]`
- Per-symbol momentum: `score = 0.5 · ret_63d + 0.5 · ret_21d` on daily closes,
  shifted by 1 day
- Count positive-score symbols and negative-score symbols
- If `max(positive, negative) ≥ alignment_threshold` (default 3 of 5) → size_mult = 1.25
- Otherwise → size_mult = 1.0
- Interpretation: "3 aligned on same side" means ≥3 symbols sharing direction, regardless of how the other 2 resolve. 3 pos / 2 neg TRIGGERS (risk-on dominant). 2 pos / 2 neg / 1 null DOES NOT trigger.

**Integration:** Layer 2 trials receive `basket_df` and compute `size_mult` per bar
(forward-filled from daily). Applied as a scalar on position size in the backtest.

### 9.2 Phase 4b — Multi-Objective Pareto + Regime Fitness

`optimize/layer2.py`:
```python
study = optuna.create_study(
    directions=["maximize", "minimize"],  # total_return_pct, max_dd_pct
    sampler=optuna.samplers.NSGAIISampler(seed=cfg.get("seed", 42)),
)
```

Trial objective returns `(total_return_pct, max_dd_pct)`.

**Selection from Pareto front:**
1. Filter front by strategy-specific max-DD cap (`max_dd_cap_pct`, default 8.0)
2. For each surviving trial, compute the **regime-specific fitness**:
   - Dominant regime for the trial = regime_state with the most bars during trades
   - **Suppressed** (`Contango_Calm`, `Neutral_Calm`):
     `fitness = (win_rate_pct ** 2) * profit_factor`
   - **Amplified** (`Backwardation_*`, `*_Elevated`):
     `fitness = (total_return_pct / max_dd_pct) * (avg_win / abs(avg_loss))`
3. Select trial with highest fitness

**Guards:**
- Zero-division on `max_dd_pct = 0` → set fitness = 0 (extremely rare, implausible)
- Zero-division on `avg_loss = 0` (all wins) → bounded by `avg_loss_min = 0.001`
- Zero trades → trial rejected before fitness is computed

Divergence gate (existing): still applied. Min-trade floor: still applied.

---

## 10. Phase 5 — Validation Suite

### 10.1 Synthetic Price-Path MC (`validation/synthetic_mc.py`)

Block bootstrap on log returns.

```python
def synthetic_price_mc(close: pd.Series, n_paths: int = 1000,
                       block_size: int = 5, seed: int = 42) -> np.ndarray:
    """
    Returns array of shape (n_paths, len(close)) of synthetic close-price paths.
    """
```

- Compute `log_returns = np.log(close).diff().dropna()`
- For each path: sample `ceil(len / block_size)` overlapping blocks with replacement
  from log_returns, concatenate, trim to length, `exp().cumprod() × close[0]`
- Preserves short-range autocorrelation; shuffles longer-range structure
- **Integration into Layer 3:** for each tuned symbol, replay the strategy on all
  1000 synthetic paths. Fail if <20% of paths produce net-profitable PnL.

### 10.2 CPCV (`validation/cpcv.py`)

```python
def cpcv_split(n_bars: int, n_blocks: int = 8, n_test_blocks: int = 2,
               purge_bars: int = 10) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yields (train_idx, test_idx) tuples. Test indices never overlap train
    indices after purging purge_bars on either side of each test block.
    """
```

- Split bars into `n_blocks` equal contiguous blocks
- For each C(n_blocks, n_test_blocks) combination, use those as test, rest as train
- Purge: drop training bars within `purge_bars` of any test block boundary
- Emits a distribution of OOS Sharpe ratios (one per fold combination)

**Integration into Layer 3:** computed once per validated symbol. Report median and
interquartile range of OOS Sharpe in HTML report.

### 10.3 DSR (`validation/dsr.py`)

Implementation of Bailey & López de Prado 2014.

```python
def deflated_sharpe_ratio(observed_sr: float, n_trials: int,
                          sr_variance: float, skew: float, kurtosis: float,
                          n_samples: int) -> float:
    """
    Returns deflated Sharpe (probability-weighted).
    """
```

Formula (Bailey-Prado 2014):
```
DSR = Z((SR - SR_0) × sqrt(T - 1) / sqrt(1 - skew × SR + (kurt - 1)/4 × SR²))
SR_0 = sqrt(V[{SR_n}]) × ((1 - γ) × Z⁻¹(1 - 1/N) + γ × Z⁻¹(1 - 1/(N·e)))
where γ is Euler-Mascheroni ≈ 0.5772
```

- `V[{SR_n}]` = variance of Sharpe ratios across all Optuna trials
- `N` = total trial count
- Headline Sharpe in HTML report becomes DSR

### 10.4 PBO (`validation/pbo.py`)

Combinatorial Symmetric Cross-Validation approach (Bailey-Prado-Borwein-Salehipour 2015).

```python
def probability_of_backtest_overfitting(is_scores: np.ndarray,
                                        oos_scores: np.ndarray) -> float:
    """
    is_scores:  (n_trials, n_folds) — in-sample score for each trial in each fold
    oos_scores: (n_trials, n_folds) — out-of-sample score for each trial in each fold
    Returns PBO in [0, 1]; values > 0.5 indicate likely overfit.
    """
```

- For each fold, rank trials by IS score; identify the IS-top trial
- Look up its OOS rank in the same fold
- Logit of (OOS rank / N); PBO = fraction of folds where logit < 0

**Integration into Layer 3:** Only computed if `cpcv_enabled=True` in config.
Reported in HTML report alongside DSR.

---

## 11. Test Strategy

### 11.1 Golden-Snapshot Regression

Written in Phase 0a, re-run after every phase:
- Legacy architecture = long-only, spot, 12 classic indicators only, EMA/ATR regime
  path
- Pipeline invoked with fixed seed + tiny budget against fixed fixtures
- Outputs byte-compared against `tests/fixtures/golden/*.json`
- Any change to the legacy path = test failure = STOP and investigate

### 11.2 Per-Module Unit Tests

- `test_indicators.py`: each indicator produces identical output on fixed fixture
- `test_regime.py`: exact boundary values (0.95, 1.02, 20/30/70/80 percentile),
  NaN handling, shift-1 correctness
- `test_fitness.py`: suppressed/amplified formulas against hand-computed expected
  values; zero-division guards
- `test_backtest_math.py`:
  - Single-trade long PnL = exit − entry
  - Single-trade short PnL = entry − exit − borrow_fee × days_held
  - FVG stop selects nearest un-filled FVG
  - ATR fallback kicks in with no FVG
- `test_synthetic_mc.py`: sign preservation on positive-drift series, variance
  scales with block_size
- `test_cpcv.py`: purge leaves no train index within `purge_bars` of any test block
- `test_dsr.py`: reproduce the numerical example from Bailey-Prado 2014 appendix
- `test_pbo.py`:
  - All-random trials → PBO near 0.5
  - All-monotonic trials → PBO near 0
  - Known-overfit synthetic → PBO > 0.5

### 11.3 Acceptance per Phase

Each phase is "done" only when:
1. Golden test passes
2. New unit tests for the phase's additions pass
3. `python apex.py --test` completes without error
4. Code committed with a phase-name-tagged commit message

---

## 12. Rollout Sequence

| # | Phase | Deliverable | Commit Gate |
|---|-------|-------------|-------------|
| 1 | 0a | Test harness + env-var config + fixtures + golden snapshot | Golden test green |
| 2 | 0b | Modularize `apex.py` → `apex/` package | Golden test green on `apex.py` shim |
| 3 | 1  | `macro_vol.py` + `options_gex.py` + `dealer_levels.py` + `six_quadrant.py` | Regime unit tests + golden still green (legacy regime path preserved) |
| 4 | 2  | `vwap_bands.py` + `vpin.py` + `vwclv.py` + `fvg.py` | Indicator unit tests + golden still green |
| 5 | 3a | Long/Short execution + borrow fees | Backtest-math tests + golden still green |
| 6 | 3c | FVG trailing stops | Stop tests + golden still green |
| 7 | 4a | Cross-asset basket | Portfolio tests + golden still green |
| 8 | 4b | Multi-objective Pareto + regime fitness | Fitness tests + legacy single-obj path preserved |
| 9 | 5  | Synthetic MC + CPCV + DSR + PBO | Validation tests + HTML report shows new metrics |

Any commit that fails a preserved-invariants gate is reverted and the work is
re-planned before re-attempting.

---

## 13. Configuration Schema Changes

Additions to `apex_config.json`:

```jsonc
{
  "borrow_rates": {
    "default": 0.02
  },

  "cross_asset_basket": {
    "symbols": ["SPY","QQQ","GLD","USO","IEF"],
    "momentum_short_days": 21,
    "momentum_long_days": 63,
    "alignment_threshold": 3,
    "size_multiplier": 1.25
  },

  "regime": {
    "vix_ratio_contango_max": 0.95,
    "vix_ratio_backwardation_min": 1.02,
    "vrp_calm_pct_low": 30,
    "vrp_calm_pct_high": 70,
    "vrp_elevated_pct_low": 20,
    "vrp_elevated_pct_high": 80,
    "vrp_rolling_window": 252
  },

  "fitness": {
    "max_dd_cap_pct": 8.0,
    "use_multi_objective": true
  },

  "validation": {
    "synthetic_mc": {
      "enabled": true,
      "n_paths": 1000,
      "block_size": 5,
      "min_profitable_pct": 20
    },
    "cpcv": {
      "enabled": true,
      "n_blocks": 8,
      "n_test_blocks": 2,
      "purge_bars": 10
    },
    "dsr": { "enabled": true },
    "pbo": { "enabled": true }
  },

  "dynamic_stops": {
    "fvg_buffer_atr_mult": 0.05,
    "atr_fallback_mult": 2.0
  }
}
```

`POLYGON_API_KEY` env var takes precedence over `polygon_api_key` in JSON.

---

## 14. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Options GEX proxy materially differs from SpotGamma's dealer model | Cache results; emit diagnostic `gex_proxy_confidence` score; allow CSV override for symbols where quality matters |
| yfinance rate-limited or discontinues VIX3M ticker | Fallback to Parkinson estimator; flag missing data; allow alternate ticker in config |
| 2-year Polygon options history caps backtest window for regime-gated strategies | Config-level `regime_required_history_days`; auto-disable regime gating for older windows |
| Synthetic MC block bootstrap destroys mean-reversion structure and rejects valid strategies | Configurable `block_size`; report pass-rate per strategy; tune threshold with empirical data |
| CPCV fold count too low for small datasets | Config-level `cpcv_min_bars`; auto-skip CPCV below threshold |
| Multi-objective Pareto front selection picks a degenerate trial | Pre-filter by min-trade floor, min-PF, max-DD cap before fitness ranking |
| Regression golden test breaks due to floating-point nondeterminism in Optuna | Pin Optuna version, pin sampler seed, set `PYTHONHASHSEED=42`, use `float32`→`float64` consistency audit |
| Large session scope (~6000 LOC) produces bugs that compound across phases | Strict per-phase commit + regression gate; stop-and-fix on any failure |

---

## 15. Out of Scope (explicit)

- Options execution (iron condor, vertical spreads, vol harvesting)
- SpotGamma subscription integration
- Futures data (ES, NQ, GC, CL, ZN)
- n8n / orchestrator integration (separate repo)
- Live broker integration beyond AFL export
- GARCH / stochastic-vol price simulation (block bootstrap only)
- Real-time execution / paper trading
- Multi-account portfolio construction
- Machine-learning classifiers beyond Optuna TPE

---

## 16. Decisions Log

| Decision | Rationale |
|----------|-----------|
| Spot-only execution, no options | User directive |
| ETF proxies for futures | Polygon Stock Starter lacks futures; SPY/QQQ/GLD/USO/IEF correlate with ES/NQ/GC/CL/ZN at 0.95+ daily |
| Block bootstrap instead of GARCH for synthetic MC | Avoids `arch` dependency; sufficient for microstructure preservation |
| VIX3M instead of VXV | CBOE discontinued VXV in 2017 |
| Compute GEX proxy from Polygon Options Starter OI/greeks | Avoids SpotGamma paid subscription; quality trade-off accepted |
| Default borrow fee 2%/yr | Industry conservative default; per-symbol override available |
| Multi-objective via NSGAIISampler | Optuna's standard multi-objective sampler; deterministic with seed |
| Regime shift-1 before merge | Prevents same-day regime signal from being used to trade that day |
| CPCV N=8, purge=10 | Balances fold count vs per-fold size; purge matches typical 1H bar trade horizon |
| Golden snapshot test frozen at Phase 0a | Every subsequent phase protected from silent drift |
| `apex.py` kept as thin shim | Preserves `python apex.py` entry point and public CLI |

---

## 17. Open Items (zero at time of approval)

All prerequisite decisions resolved. Implementation plan is next.
