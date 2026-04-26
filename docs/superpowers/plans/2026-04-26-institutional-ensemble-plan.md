# Institutional Multi-Strategy Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 6-strategy structural-primitives ensemble (VRP+GEX fade, OPEX gravity, VIX term structure, vol skew arb, SMC structural, cross-asset vol overlay) on top of the existing Optuna Screener pipeline. Combine via risk parity + regime overlay. Validate per-strategy and at portfolio level via CPCV. Realistic ensemble Sharpe target: 1.0-1.5.

**Architecture:** Each strategy is an independent `Strategy` class under `apex/strategies/`, implementing a common interface (compute_signals, compute_position_size, get_tunable_params, get_data_requirements). The ensemble combiner under `apex/ensemble/` runs all strategies, computes risk-parity weights from rolling 60-day vol, applies regime tilts, caps single-strategy concentration at 30%, and emits a final portfolio NAV. Validation happens at three layers: per-strategy CPCV (Layer A), ensemble portfolio CPCV (Layer B), walk-forward weight stability (Layer C).

**Tech Stack:** Python 3.11+, Optuna (TPE + NSGA-II), pandas, numpy, pyarrow, scipy, requests, yfinance, plotly, pytest, pytest-xdist. Polygon.io Stock Starter + Options Starter, FRED API.

**Spec:** `docs/superpowers/specs/2026-04-26-institutional-ensemble-design.md`

---

## File Structure (target)

```
apex/
  strategies/                              NEW package
    __init__.py                            STRATEGY_REGISTRY = {name: cls}
    base.py                                StrategyBase abstract class
    vrp_gex_fade.py                        Strategy 1
    opex_gravity.py                        Strategy 2
    vix_term_structure.py                  Strategy 3
    vol_skew_arb.py                        Strategy 4
    smc_structural.py                      Strategy 5
    cross_asset_vol_overlay.py             Strategy 6 (overlay)

  ensemble/                                NEW package
    __init__.py
    risk_parity.py                         compute_risk_parity_weights()
    regime_overlay.py                      apply_regime_tilts()
    combiner.py                            EnsembleCombiner orchestrator

  data/
    options_chain.py                       NEW: full chain helpers (skew, OPEX scan)
    cross_asset_vol.py                     NEW: MOVE + OVX fetchers via FRED
    macro_vol.py                           EXTEND: fetch_macro_volatility (existing)

  validation/
    ensemble_cpcv.py                       NEW: portfolio-level CPCV
    walk_forward.py                        NEW: weight stability validation

  main.py                                  EXTEND: --ensemble flag
  report/
    ensemble_report.py                     NEW: ensemble-specific HTML report

tests/
  test_strategy_base.py                    interface contract
  test_strategy_vrp_gex.py
  test_strategy_opex.py
  test_strategy_vix_term.py
  test_strategy_vol_skew.py
  test_strategy_smc.py
  test_strategy_overlay.py
  test_ensemble_risk_parity.py
  test_ensemble_regime_overlay.py
  test_ensemble_combiner.py
  test_ensemble_cpcv.py
  test_walk_forward.py
  test_data_options_chain.py
  test_data_cross_asset_vol.py
```

---

## Global Conventions

- **Python version:** 3.11+
- **Test runner:** `pytest -v` from repo root
- **Float comparisons:** `pytest.approx(x, abs=1e-9)` unless looser specified
- **Seed discipline:** all tests set `np.random.seed(42)`; Optuna uses `TPESampler(seed=42)` or `NSGAIISampler(seed=42)`
- **Commit message format:** `phase-12<step>: <summary>` (e.g., `phase-12a: add StrategyBase interface`)
- **Phase gate:** after every task, `pytest tests/test_regression_golden.py` MUST pass — legacy pipeline byte-equal preserved
- **Do NOT push** to origin until each phase's gate passes
- **API keys:** `POLYGON_API_KEY` and `FRED_API_KEY` from `.env` (gitignored)

---

## Phase Quick Index

| Phase | Tasks | Deliverable |
|-------|-------|-------------|
| 12A | 1-10 | Framework: StrategyBase + ensemble package + ensemble CPCV scaffold |
| 12B | 11-20 | Strategy 1: VRP+GEX fade enhanced |
| 12C | 21-32 | Strategy 2: OPEX gravity |
| 12D | 33-42 | Strategy 3: VIX term structure |
| 12E | 43-56 | Strategy 4: Vol skew arbitrage |
| 12F | 57-68 | Strategy 5: SMC structural |
| 12G | 69-74 | Strategy 6: Cross-asset vol overlay |
| 12H | 75-86 | Ensemble integration + Layers B/C validation |
| 12I | 87-92 | HTML report + CLI integration + final E2E |

---

## Phase 12A — Framework

### Task 1: Create strategies package skeleton

**Files:**
- Create: `apex/strategies/__init__.py`
- Create: `tests/test_strategies_package.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_strategies_package.py`:

```python
"""Sanity check: apex.strategies package importable with empty registry."""
def test_strategies_package_importable():
    import apex.strategies
    assert hasattr(apex.strategies, "STRATEGY_REGISTRY")
    assert isinstance(apex.strategies.STRATEGY_REGISTRY, dict)
```

- [ ] **Step 2: Run — fails (no module)**

Run: `pytest tests/test_strategies_package.py -v`
Expected: ModuleNotFoundError on `apex.strategies`.

- [ ] **Step 3: Create the package**

Create `apex/strategies/__init__.py`:

```python
"""Strategy modules for the institutional multi-strategy ensemble.

Each strategy implements StrategyBase (see base.py). Strategies register
themselves via @register_strategy decorator on their class.
"""
STRATEGY_REGISTRY: dict = {}


def register_strategy(cls):
    """Class decorator that registers a strategy in STRATEGY_REGISTRY."""
    if not hasattr(cls, "name"):
        raise TypeError(f"{cls.__name__} must define a class attribute `name`")
    STRATEGY_REGISTRY[cls.name] = cls
    return cls
```

- [ ] **Step 4: Run — passes**

Run: `pytest tests/test_strategies_package.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apex/strategies/__init__.py tests/test_strategies_package.py
git commit -m "phase-12a: create strategies package skeleton with STRATEGY_REGISTRY"
```

---

### Task 2: StrategyBase abstract class

**Files:**
- Create: `apex/strategies/base.py`
- Create: `tests/test_strategy_base.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_strategy_base.py`:

```python
"""Tests for StrategyBase abstract interface."""
import pytest


def test_strategybase_cannot_be_instantiated_directly():
    from apex.strategies.base import StrategyBase
    with pytest.raises(TypeError):
        StrategyBase()


def test_concrete_strategy_must_implement_compute_signals():
    from apex.strategies.base import StrategyBase

    class IncompleteStrategy(StrategyBase):
        name = "incomplete"
        data_requirements = []
        # missing compute_signals
    with pytest.raises(TypeError):
        IncompleteStrategy()


def test_concrete_strategy_works_when_complete():
    from apex.strategies.base import StrategyBase
    import pandas as pd

    class MyStrategy(StrategyBase):
        name = "test_my_strategy"
        data_requirements = ["exec_df_1H"]

        def compute_signals(self, data):
            n = len(data["exec_df_1H"])
            return pd.DataFrame({
                "entry_long": [False] * n,
                "entry_short": [False] * n,
                "exit_long": [False] * n,
                "exit_short": [False] * n,
            })

        def compute_position_size(self, data, signals):
            return pd.Series([0.0] * len(signals))

        def get_tunable_params(self):
            return {}

    s = MyStrategy()
    assert s.name == "test_my_strategy"
    assert s.data_requirements == ["exec_df_1H"]


def test_register_strategy_decorator_adds_to_registry():
    from apex.strategies import STRATEGY_REGISTRY, register_strategy
    from apex.strategies.base import StrategyBase
    import pandas as pd

    @register_strategy
    class RegisteredStrategy(StrategyBase):
        name = "test_registered_strategy"
        data_requirements = []

        def compute_signals(self, data):
            return pd.DataFrame()

        def compute_position_size(self, data, signals):
            return pd.Series()

        def get_tunable_params(self):
            return {}

    assert "test_registered_strategy" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["test_registered_strategy"] is RegisteredStrategy
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/test_strategy_base.py -v`
Expected: ModuleNotFoundError on `apex.strategies.base`.

- [ ] **Step 3: Implement apex/strategies/base.py**

Create `apex/strategies/base.py`:

```python
"""Abstract base class that every strategy in the ensemble must implement."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class StrategyBase(ABC):
    """Each strategy in the ensemble subclasses this.

    Subclasses MUST set:
      name: str               — short identifier (e.g., "vrp_gex_fade")
      data_requirements: list — keys the strategy expects in the `data` dict
                                passed to compute_signals (e.g., "exec_df_1H",
                                "options_chain_daily", "vix").

    Subclasses MUST implement:
      compute_signals(data) -> DataFrame with columns:
          entry_long, entry_short, exit_long, exit_short (all bool/int Series).
      compute_position_size(data, signals) -> Series of position sizes
          in [-1.0, +1.0]. The ensemble combiner scales these via risk parity.
      get_tunable_params() -> dict[param_name, (lo, hi, type)]
          Optuna search space for this strategy's tunable parameters.
    """

    name: str = ""
    data_requirements: List[str] = []

    @abstractmethod
    def compute_signals(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Return DataFrame with bool columns: entry_long/short, exit_long/short."""

    @abstractmethod
    def compute_position_size(self, data: Dict[str, Any],
                              signals: pd.DataFrame) -> pd.Series:
        """Return per-bar position size in [-1.0, +1.0]."""

    @abstractmethod
    def get_tunable_params(self) -> Dict[str, tuple]:
        """Optuna search space. Returns {param: (lo, hi, type)}.
        type is 'int', 'float', or 'categorical' (with options as 3rd tuple element)."""
```

- [ ] **Step 4: Run — passes**

Run: `pytest tests/test_strategy_base.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apex/strategies/base.py tests/test_strategy_base.py
git commit -m "phase-12a: StrategyBase abstract interface + register_strategy decorator"
```

---

### Task 3: Ensemble package skeleton

**Files:**
- Create: `apex/ensemble/__init__.py`
- Create: `tests/test_ensemble_package.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_ensemble_package.py`:

```python
"""Sanity check: apex.ensemble package importable."""
def test_ensemble_package_importable():
    import apex.ensemble
    assert apex.ensemble.__name__ == "apex.ensemble"
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/test_ensemble_package.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Create the package**

Create `apex/ensemble/__init__.py`:

```python
"""Multi-strategy ensemble combiner.

Combines outputs from individual strategies (under apex.strategies) via
risk-parity weights with regime-conditional overlay. See combiner.py for
the orchestrator.
"""
```

- [ ] **Step 4: Run — passes**

Run: `pytest tests/test_ensemble_package.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apex/ensemble/__init__.py tests/test_ensemble_package.py
git commit -m "phase-12a: create ensemble package skeleton"
```

---

### Task 4: Risk parity weight computation

**Files:**
- Create: `apex/ensemble/risk_parity.py`
- Create: `tests/test_ensemble_risk_parity.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ensemble_risk_parity.py`:

```python
"""Tests for risk-parity weight computation."""
import numpy as np
import pandas as pd
import pytest


def test_equal_vol_strategies_get_equal_weights():
    from apex.ensemble.risk_parity import compute_risk_parity_weights
    # Three strategies with identical vol → equal weights
    rng = np.random.default_rng(42)
    returns = {
        "s1": pd.Series(rng.normal(0, 0.01, 60)),
        "s2": pd.Series(rng.normal(0, 0.01, 60)),
        "s3": pd.Series(rng.normal(0, 0.01, 60)),
    }
    weights = compute_risk_parity_weights(returns)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)
    for w in weights.values():
        assert abs(w - 1/3) < 0.05  # within 5% of equal


def test_higher_vol_strategy_gets_lower_weight():
    from apex.ensemble.risk_parity import compute_risk_parity_weights
    rng = np.random.default_rng(0)
    returns = {
        "low_vol":  pd.Series(rng.normal(0, 0.005, 60)),
        "high_vol": pd.Series(rng.normal(0, 0.020, 60)),
    }
    weights = compute_risk_parity_weights(returns)
    assert weights["low_vol"] > weights["high_vol"]
    # Low-vol weight ≈ 4x high-vol weight (vol ratio is 4)
    assert weights["low_vol"] / weights["high_vol"] == pytest.approx(4.0, rel=0.20)


def test_zero_vol_strategy_gets_zero_weight():
    from apex.ensemble.risk_parity import compute_risk_parity_weights
    returns = {
        "active": pd.Series(np.random.default_rng(1).normal(0, 0.01, 60)),
        "dead": pd.Series([0.0] * 60),  # zero vol
    }
    weights = compute_risk_parity_weights(returns)
    assert weights["dead"] == 0.0
    assert weights["active"] == pytest.approx(1.0)


def test_weights_sum_to_one():
    from apex.ensemble.risk_parity import compute_risk_parity_weights
    rng = np.random.default_rng(7)
    returns = {f"s{i}": pd.Series(rng.normal(0, 0.005 + i*0.003, 60))
               for i in range(5)}
    weights = compute_risk_parity_weights(returns)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)


def test_max_weight_cap_enforced():
    """When one strategy has zero vol, others split — but no single strategy
    should ever exceed the max_weight cap."""
    from apex.ensemble.risk_parity import compute_risk_parity_weights
    returns = {
        "tiny_vol": pd.Series([0.001] * 60),
        "high_vol_a": pd.Series(np.random.default_rng(2).normal(0, 0.05, 60)),
        "high_vol_b": pd.Series(np.random.default_rng(3).normal(0, 0.05, 60)),
    }
    weights = compute_risk_parity_weights(returns, max_weight=0.30)
    for w in weights.values():
        assert w <= 0.30 + 1e-9
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/test_ensemble_risk_parity.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement apex/ensemble/risk_parity.py**

Create `apex/ensemble/risk_parity.py`:

```python
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
```

- [ ] **Step 4: Run — passes**

Run: `pytest tests/test_ensemble_risk_parity.py -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apex/ensemble/risk_parity.py tests/test_ensemble_risk_parity.py
git commit -m "phase-12a: risk parity weight computation with max-weight cap"
```

---

### Task 5: Regime overlay tilts

**Files:**
- Create: `apex/ensemble/regime_overlay.py`
- Create: `tests/test_ensemble_regime_overlay.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ensemble_regime_overlay.py`:

```python
"""Tests for regime-conditional weight tilts."""
import pytest


MEAN_REV_STRATEGIES = ("vrp_gex_fade", "vol_skew_arb", "smc_structural")
TREND_STRATEGIES = ("opex_gravity", "vix_term_structure")


def test_r1_boosts_mean_reversion_strategies():
    from apex.ensemble.regime_overlay import apply_regime_tilts
    base = {"vrp_gex_fade": 0.2, "opex_gravity": 0.2, "vix_term_structure": 0.2,
            "vol_skew_arb": 0.2, "smc_structural": 0.2}
    tilted = apply_regime_tilts(base, regime="R1")
    # Mean-rev strategies should have higher weight than trend strategies
    for mr in MEAN_REV_STRATEGIES:
        for tr in TREND_STRATEGIES:
            assert tilted[mr] > tilted[tr]
    assert sum(tilted.values()) == pytest.approx(1.0, abs=1e-9)


def test_r3_boosts_trend_strategies():
    from apex.ensemble.regime_overlay import apply_regime_tilts
    base = {"vrp_gex_fade": 0.2, "opex_gravity": 0.2, "vix_term_structure": 0.2,
            "vol_skew_arb": 0.2, "smc_structural": 0.2}
    tilted = apply_regime_tilts(base, regime="R3")
    for tr in TREND_STRATEGIES:
        for mr in MEAN_REV_STRATEGIES:
            assert tilted[tr] > tilted[mr]
    assert sum(tilted.values()) == pytest.approx(1.0, abs=1e-9)


def test_r4_zeros_all_weights():
    from apex.ensemble.regime_overlay import apply_regime_tilts
    base = {"vrp_gex_fade": 0.2, "opex_gravity": 0.2, "vix_term_structure": 0.2,
            "vol_skew_arb": 0.2, "smc_structural": 0.2}
    tilted = apply_regime_tilts(base, regime="R4")
    for w in tilted.values():
        assert w == 0.0


def test_unknown_regime_returns_unchanged():
    from apex.ensemble.regime_overlay import apply_regime_tilts
    base = {"vrp_gex_fade": 0.5, "opex_gravity": 0.5}
    tilted = apply_regime_tilts(base, regime="UNKNOWN")
    assert tilted == base


def test_missing_strategy_in_base_is_handled():
    """Tilts should only apply to strategies actually present in the weights."""
    from apex.ensemble.regime_overlay import apply_regime_tilts
    base = {"vrp_gex_fade": 0.5, "opex_gravity": 0.5}
    tilted = apply_regime_tilts(base, regime="R1")
    assert "vrp_gex_fade" in tilted and "opex_gravity" in tilted
    assert sum(tilted.values()) == pytest.approx(1.0, abs=1e-9)
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/test_ensemble_regime_overlay.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement apex/ensemble/regime_overlay.py**

Create `apex/ensemble/regime_overlay.py`:

```python
"""Regime-conditional weight tilts for the ensemble."""
from typing import Dict


MEAN_REVERSION_STRATEGIES = {"vrp_gex_fade", "vol_skew_arb", "smc_structural"}
TREND_STRATEGIES = {"opex_gravity", "vix_term_structure"}

TILT_FACTOR = 1.20  # 20% boost to favored strategies in current regime


def apply_regime_tilts(weights: Dict[str, float],
                        regime: str) -> Dict[str, float]:
    """Apply regime-conditional multiplicative tilts and renormalize.

    Suppressed regimes (R1, R2, Contango_Calm, Neutral_Calm):
        boost mean-reversion strategies by TILT_FACTOR.
    Amplified regimes (R3, Backwardation, Elevated VRP):
        boost trend-following strategies by TILT_FACTOR.
    R4 (no-trade): all weights → 0.
    Unknown regime: weights returned unchanged.
    """
    if regime == "R4":
        return {name: 0.0 for name in weights}

    if regime in {"R1", "R2", "Contango_Calm", "Neutral_Calm"}:
        boost_set = MEAN_REVERSION_STRATEGIES
    elif regime in {"R3", "Contango_Elevated", "Neutral_Elevated",
                    "Backwardation_Calm", "Backwardation_Elevated"}:
        boost_set = TREND_STRATEGIES
    else:
        return dict(weights)

    tilted = {}
    for name, w in weights.items():
        if name in boost_set:
            tilted[name] = w * TILT_FACTOR
        else:
            tilted[name] = w

    total = sum(tilted.values())
    if total <= 1e-9:
        return tilted
    return {name: w / total for name, w in tilted.items()}
```

- [ ] **Step 4: Run — passes**

Run: `pytest tests/test_ensemble_regime_overlay.py -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apex/ensemble/regime_overlay.py tests/test_ensemble_regime_overlay.py
git commit -m "phase-12a: regime-conditional weight tilts (R1/R2 mean-rev, R3 trend, R4 zero)"
```

---

### Task 6: EnsembleCombiner orchestrator

**Files:**
- Create: `apex/ensemble/combiner.py`
- Create: `tests/test_ensemble_combiner.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ensemble_combiner.py`:

```python
"""Tests for EnsembleCombiner orchestrator."""
import numpy as np
import pandas as pd
import pytest


class _FakeStrategy:
    """Minimal strategy for combiner tests — fixed signals + size."""
    def __init__(self, name, signal_series, size_value=0.5):
        self.name = name
        self.data_requirements = []
        self._signal = signal_series
        self._size = size_value

    def compute_signals(self, data):
        n = len(self._signal)
        return pd.DataFrame({
            "entry_long":  self._signal.astype(bool),
            "entry_short": pd.Series([False] * n),
            "exit_long":   pd.Series([False] * n),
            "exit_short":  pd.Series([False] * n),
        })

    def compute_position_size(self, data, signals):
        return pd.Series([self._size] * len(self._signal))

    def get_tunable_params(self):
        return {}


def _make_data(n=100):
    return {
        "exec_df_1H": pd.DataFrame({
            "datetime": pd.date_range("2025-01-02", periods=n, freq="h"),
            "close": np.linspace(100, 105, n),
        }),
        "regime_state": pd.Series(["R2"] * n),
    }


def test_combiner_runs_with_two_strategies():
    from apex.ensemble.combiner import EnsembleCombiner
    n = 100
    sig_a = pd.Series([True if i % 10 == 0 else False for i in range(n)])
    sig_b = pd.Series([True if i % 7 == 0 else False for i in range(n)])
    strategies = [_FakeStrategy("vrp_gex_fade", sig_a, 0.5),
                  _FakeStrategy("opex_gravity", sig_b, 0.4)]
    combiner = EnsembleCombiner(strategies)
    result = combiner.run(_make_data(n))
    assert "portfolio_position" in result
    assert "weights" in result
    assert "trades" in result
    assert isinstance(result["portfolio_position"], pd.Series)
    assert len(result["portfolio_position"]) == n


def test_combiner_respects_30pct_cap():
    """No strategy should be allocated more than 30% of portfolio weight."""
    from apex.ensemble.combiner import EnsembleCombiner
    n = 100
    # All strategies same vol → equal weights would exceed cap if only 2 strategies
    strategies = [_FakeStrategy(f"vrp_gex_fade", pd.Series([False] * n), 0.5),
                  _FakeStrategy(f"opex_gravity", pd.Series([False] * n), 0.5)]
    combiner = EnsembleCombiner(strategies, max_weight=0.30)
    result = combiner.run(_make_data(n))
    for w in result["weights"].values():
        assert w <= 0.30 + 1e-9


def test_combiner_zeroes_weights_in_r4():
    from apex.ensemble.combiner import EnsembleCombiner
    n = 50
    strategies = [_FakeStrategy("vrp_gex_fade",
                                pd.Series([True] * n), 0.5)]
    data = _make_data(n)
    data["regime_state"] = pd.Series(["R4"] * n)
    combiner = EnsembleCombiner(strategies)
    result = combiner.run(data)
    # Final positions should be zero in R4
    assert (result["portfolio_position"].abs() < 1e-9).all()
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/test_ensemble_combiner.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement apex/ensemble/combiner.py**

Create `apex/ensemble/combiner.py`:

```python
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
        """
        per_strategy_signals: Dict[str, pd.DataFrame] = {}
        per_strategy_positions: Dict[str, pd.Series] = {}
        per_strategy_returns: Dict[str, pd.Series] = {}

        for s in self.strategies:
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

        # Combine per-strategy positions
        n = len(next(iter(per_strategy_positions.values())))
        combined = pd.Series([0.0] * n)
        for name, pos in per_strategy_positions.items():
            w = weights.get(name, 0.0)
            combined = combined + w * pos.values

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
```

- [ ] **Step 4: Run — passes**

Run: `pytest tests/test_ensemble_combiner.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apex/ensemble/combiner.py tests/test_ensemble_combiner.py
git commit -m "phase-12a: EnsembleCombiner orchestrator (risk parity + regime overlay)"
```

---

### Task 7: Ensemble CPCV (portfolio-level)

**Files:**
- Create: `apex/validation/ensemble_cpcv.py`
- Create: `tests/test_ensemble_cpcv.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ensemble_cpcv.py`:

```python
"""Tests for portfolio-level ensemble CPCV."""
import numpy as np
import pandas as pd
import pytest


def test_ensemble_cpcv_returns_sharpe_distribution():
    from apex.validation.ensemble_cpcv import evaluate_ensemble_cpcv
    rng = np.random.default_rng(42)
    n = 1000
    # Synthetic portfolio NAV: positive drift + noise
    portfolio_returns = pd.Series(rng.normal(0.0005, 0.01, n))
    result = evaluate_ensemble_cpcv(portfolio_returns,
                                     n_blocks=4, n_test_blocks=1)
    assert result["n_folds"] > 0
    assert "sharpe_median" in result
    assert "sharpe_iqr" in result
    assert isinstance(result["oos_sharpes"], list)


def test_ensemble_cpcv_handles_empty_returns():
    from apex.validation.ensemble_cpcv import evaluate_ensemble_cpcv
    result = evaluate_ensemble_cpcv(pd.Series([]),
                                     n_blocks=4, n_test_blocks=1)
    assert result.get("n_folds", 0) == 0


def test_ensemble_cpcv_positive_drift_yields_positive_median():
    """Strong positive drift → median Sharpe should be positive."""
    from apex.validation.ensemble_cpcv import evaluate_ensemble_cpcv
    rng = np.random.default_rng(0)
    # Strong drift, small noise
    returns = pd.Series(rng.normal(0.005, 0.005, 800))
    result = evaluate_ensemble_cpcv(returns, n_blocks=4, n_test_blocks=1)
    assert result["sharpe_median"] > 0
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/test_ensemble_cpcv.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement apex/validation/ensemble_cpcv.py**

Create `apex/validation/ensemble_cpcv.py`:

```python
"""Portfolio-level CPCV evaluation for ensemble NAV."""
from typing import Any, Dict

import numpy as np
import pandas as pd

from apex.validation.cpcv import cpcv_split


def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe assuming returns are per-bar."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma <= 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(periods_per_year))


def evaluate_ensemble_cpcv(portfolio_returns: pd.Series,
                            n_blocks: int = 8,
                            n_test_blocks: int = 2,
                            purge_bars: int = 10,
                            periods_per_year: int = 252) -> Dict[str, Any]:
    """Run CPCV at the portfolio NAV level.

    Args:
        portfolio_returns: Series of per-bar portfolio returns
        n_blocks, n_test_blocks, purge_bars: passed to cpcv_split

    Returns:
        {n_folds, oos_sharpes, sharpe_median, sharpe_iqr, sharpe_pct_positive,
         oos_returns}
    """
    n = len(portfolio_returns)
    if n < 100:
        return {"n_folds": 0, "error": "insufficient bars"}

    sharpes = []
    cum_returns = []
    for train_idx, test_idx in cpcv_split(n, n_blocks=n_blocks,
                                           n_test_blocks=n_test_blocks,
                                           purge_bars=purge_bars):
        if len(test_idx) < 30:
            continue
        test_returns = portfolio_returns.iloc[test_idx]
        s = _annualized_sharpe(test_returns, periods_per_year)
        sharpes.append(s)
        cum_returns.append(float((1 + test_returns).prod() - 1))

    if not sharpes:
        return {"n_folds": 0, "error": "no successful folds"}

    arr = np.array(sharpes)
    return {
        "n_folds": len(sharpes),
        "oos_sharpes": sharpes,
        "oos_returns": cum_returns,
        "sharpe_median": float(np.median(arr)),
        "sharpe_iqr": (float(np.percentile(arr, 25)),
                       float(np.percentile(arr, 75))),
        "sharpe_pct_positive": float((arr > 0).mean()),
    }
```

- [ ] **Step 4: Run — passes**

Run: `pytest tests/test_ensemble_cpcv.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apex/validation/ensemble_cpcv.py tests/test_ensemble_cpcv.py
git commit -m "phase-12a: portfolio-level ensemble CPCV evaluator"
```

---

### Task 8: Walk-forward weight stability

**Files:**
- Create: `apex/validation/walk_forward.py`
- Create: `tests/test_walk_forward.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_walk_forward.py`:

```python
"""Tests for walk-forward weight validation."""
import numpy as np
import pandas as pd
import pytest


def test_walk_forward_compares_dynamic_vs_static():
    from apex.validation.walk_forward import compare_dynamic_vs_static_weights
    rng = np.random.default_rng(42)
    n_months = 24
    # Strategy A: stable positive Sharpe; Strategy B: noisy
    months = pd.date_range("2023-01-01", periods=n_months, freq="ME")
    monthly_returns = {
        "vrp_gex_fade": pd.Series(rng.normal(0.01, 0.02, n_months), index=months),
        "opex_gravity": pd.Series(rng.normal(0.005, 0.05, n_months), index=months),
    }
    result = compare_dynamic_vs_static_weights(monthly_returns)
    assert "static_sharpe" in result
    assert "dynamic_sharpe" in result
    assert "uplift" in result
    assert isinstance(result["uplift"], float)


def test_walk_forward_handles_short_history():
    from apex.validation.walk_forward import compare_dynamic_vs_static_weights
    monthly_returns = {
        "vrp_gex_fade": pd.Series([0.01, -0.02]),
        "opex_gravity": pd.Series([0.005, 0.001]),
    }
    result = compare_dynamic_vs_static_weights(monthly_returns)
    assert result.get("n_months", 0) <= 2 or "error" in result
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/test_walk_forward.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement apex/validation/walk_forward.py**

Create `apex/validation/walk_forward.py`:

```python
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
```

- [ ] **Step 4: Run — passes**

Run: `pytest tests/test_walk_forward.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apex/validation/walk_forward.py tests/test_walk_forward.py
git commit -m "phase-12a: walk-forward weight validation (dynamic vs static)"
```

---

### Task 9: Phase 12A regression gate

**Files:** none

- [ ] **Step 1: Run full suite**

Run: `python -m pytest tests/ -v`
Expected: ALL existing 173 tests + ~17 new framework tests PASS (~190 total). Golden snapshot still byte-equal.

- [ ] **Step 2: Verify framework imports work**

Run:
```bash
python -c "
from apex.strategies.base import StrategyBase
from apex.strategies import STRATEGY_REGISTRY, register_strategy
from apex.ensemble.risk_parity import compute_risk_parity_weights
from apex.ensemble.regime_overlay import apply_regime_tilts
from apex.ensemble.combiner import EnsembleCombiner
from apex.validation.ensemble_cpcv import evaluate_ensemble_cpcv
from apex.validation.walk_forward import compare_dynamic_vs_static_weights
print('All framework imports OK')
"
```
Expected: prints `All framework imports OK`.

- [ ] **Step 3: Tag phase complete**

Phase 12A framework done. Strategies 1-6 plug into this.

---

### Task 10: Phase 12A completion marker

(informational only — no actions)

Phase 12A delivered: StrategyBase interface, ensemble package (risk parity + regime overlay + combiner), portfolio-level CPCV, walk-forward weight validation. All tests green.

---

## Phase 12B — Strategy 1: VRP+GEX Fade

**Strategy 1 enhances the existing VRP fade with real options-derived gamma walls.** Reuses `compute_gex_proxy` (Phase 1), `compute_vrp_regime` (Phase 1), `compute_vpin` (Phase 2), `compute_rsi` (legacy basics), `detect_breakout_reversal` (Phase 6).

---

### Task 11: Data fetcher for VRP+GEX strategy

**Files:**
- Create: `apex/data/options_chain.py`
- Create: `tests/test_data_options_chain.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_data_options_chain.py`:

```python
"""Tests for options-chain helpers used by Strategy 1 + Strategy 4."""
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def mock_chain():
    """Synthetic options chain — same shape as options_chain_sample.json fixture."""
    path = Path(__file__).parent / "fixtures" / "options_chain_sample.json"
    return json.loads(path.read_text())


def test_extract_call_put_walls(mock_chain, monkeypatch):
    from apex.data import options_chain
    monkeypatch.setattr(options_chain, "_fetch_chain_for_date",
                         lambda sym, dt, cd: mock_chain)
    levels = options_chain.fetch_gex_levels("SPY", "2025-06-17", cache_dir=None)
    assert "call_wall" in levels
    assert "put_wall" in levels
    assert "gamma_flip" in levels
    assert isinstance(levels["call_wall"], float)
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/test_data_options_chain.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement apex/data/options_chain.py**

Create `apex/data/options_chain.py`:

```python
"""Options-chain helpers for strategies 1 (gamma walls) and 4 (vol skew)."""
from pathlib import Path
from typing import Optional

from apex.data.options_gex import compute_gex_proxy, _fetch_chain as _fetch_chain_for_date


def fetch_gex_levels(symbol: str, as_of: str, cache_dir: Optional[Path]) -> dict:
    """Wrapper around compute_gex_proxy that returns a clean dict for strategy use.

    Returns the same shape as compute_gex_proxy: call_wall, put_wall, gamma_flip,
    vol_trigger, abs_gamma_strike.
    """
    return compute_gex_proxy(symbol, as_of, cache_dir)
```

- [ ] **Step 4: Run — passes**

Run: `pytest tests/test_data_options_chain.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apex/data/options_chain.py tests/test_data_options_chain.py
git commit -m "phase-12b: options_chain helper for gamma-wall extraction"
```

---

### Task 12: VRP+GEX Fade strategy class

**Files:**
- Create: `apex/strategies/vrp_gex_fade.py`
- Create: `tests/test_strategy_vrp_gex.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_strategy_vrp_gex.py`:

```python
"""Tests for Strategy 1: VRP+GEX Fade."""
import numpy as np
import pandas as pd
import pytest


def _make_test_data(n=200, regime="R1"):
    """Build minimal data dict with VRP regime + gamma walls."""
    rng = np.random.default_rng(42)
    close = 400 + np.cumsum(rng.normal(0, 0.5, n))
    return {
        "exec_df_1H": pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:30", periods=n, freq="h"),
            "open": close - 0.05, "high": close + 0.20,
            "low": close - 0.20, "close": close,
            "volume": rng.integers(10000, 50000, n).astype(float),
            "vix": np.full(n, 14.0),
            "vxv": np.full(n, 17.0),
            "vrp_pct": np.full(n, 85.0),
            # Strategy 1 needs gamma wall columns
            "call_wall": np.full(n, 410.0),
            "put_wall": np.full(n, 390.0),
            "gamma_flip": np.full(n, 400.0),
        }),
        "regime_state": pd.Series([regime] * n),
        "symbol": "SPY",
    }


def test_strategy_registers():
    """Strategy 1 should auto-register in STRATEGY_REGISTRY."""
    from apex.strategies import STRATEGY_REGISTRY
    from apex.strategies import vrp_gex_fade  # triggers @register_strategy
    assert "vrp_gex_fade" in STRATEGY_REGISTRY


def test_compute_signals_returns_correct_columns():
    from apex.strategies.vrp_gex_fade import VRPGEXFadeStrategy
    s = VRPGEXFadeStrategy()
    data = _make_test_data()
    signals = s.compute_signals(data)
    for c in ("entry_long", "entry_short", "exit_long", "exit_short"):
        assert c in signals.columns
    assert len(signals) == len(data["exec_df_1H"])


def test_no_entries_in_r4_regime():
    from apex.strategies.vrp_gex_fade import VRPGEXFadeStrategy
    s = VRPGEXFadeStrategy()
    data = _make_test_data(regime="R4")
    signals = s.compute_signals(data)
    assert not signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_tunable_params_match_spec():
    """Spec lists 7 tunable params for Strategy 1."""
    from apex.strategies.vrp_gex_fade import VRPGEXFadeStrategy
    s = VRPGEXFadeStrategy()
    params = s.get_tunable_params()
    expected = {"vrp_pct_threshold", "gamma_wall_proximity_atr",
                "rsi2_oversold", "rsi2_overbought",
                "vpin_pct_max", "stop_atr_mult", "max_hold_bars"}
    assert set(params.keys()) == expected
```

- [ ] **Step 2: Run — fails**

Run: `pytest tests/test_strategy_vrp_gex.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement apex/strategies/vrp_gex_fade.py**

Create `apex/strategies/vrp_gex_fade.py`:

```python
"""Strategy 1: VRP + GEX Fade.

Structural primitive: Real options-derived gamma walls (Call Wall / Put Wall)
combined with VRP percentile filter. Trades mean-reversion in suppressed
volatility regimes when price is near a gamma wall.

Spec: docs/superpowers/specs/2026-04-26-institutional-ensemble-design.md §5.1
"""
from typing import Any, Dict

import numpy as np
import pandas as pd

from apex.indicators.basics import compute_atr, compute_rsi
from apex.indicators.vpin import compute_vpin
from apex.strategies import register_strategy
from apex.strategies.base import StrategyBase


@register_strategy
class VRPGEXFadeStrategy(StrategyBase):
    name = "vrp_gex_fade"
    data_requirements = ["exec_df_1H", "regime_state"]

    # Strategy 1 also needs gamma walls (call_wall, put_wall columns) on
    # exec_df_1H. The ensemble pre-merges these via apex.data.dealer_levels.

    def __init__(self, params: Dict[str, Any] | None = None):
        defaults = {
            "vrp_pct_threshold": 70,
            "gamma_wall_proximity_atr": 0.5,
            "rsi2_oversold": 15,
            "rsi2_overbought": 85,
            "vpin_pct_max": 50,
            "stop_atr_mult": 1.0,
            "max_hold_bars": 21,
        }
        if params:
            defaults.update(params)
        self.params = defaults

    def compute_signals(self, data: Dict[str, Any]) -> pd.DataFrame:
        df = data["exec_df_1H"]
        regime = data["regime_state"] if "regime_state" in data else pd.Series(
            ["R4"] * len(df))

        n = len(df)
        entry_long = np.zeros(n, dtype=bool)
        entry_short = np.zeros(n, dtype=bool)
        exit_long = np.zeros(n, dtype=bool)
        exit_short = np.zeros(n, dtype=bool)

        # Pre-compute features
        atr = compute_atr(df, period=14).values
        rsi2 = compute_rsi(df["close"], period=2).values
        vpin_df = compute_vpin(df)
        vpin_pct = vpin_df["vpin_pct"].values

        vrp_pct = df.get("vrp_pct")
        vix = df.get("vix")
        vxv = df.get("vxv")
        call_wall = df.get("call_wall")
        put_wall = df.get("put_wall")

        if any(s is None for s in (vrp_pct, vix, vxv, call_wall, put_wall)):
            return pd.DataFrame({
                "entry_long": entry_long, "entry_short": entry_short,
                "exit_long": exit_long, "exit_short": exit_short,
            })

        for i in range(n):
            if regime.iloc[i] == "R4":
                continue
            if pd.isna(vrp_pct.iloc[i]) or pd.isna(vix.iloc[i]) or pd.isna(vxv.iloc[i]):
                continue
            ts_ratio = vix.iloc[i] / vxv.iloc[i] if vxv.iloc[i] > 0 else float("inf")

            # Filters: suppressed regime + contango + low vol
            if vrp_pct.iloc[i] < self.params["vrp_pct_threshold"]:
                continue
            if ts_ratio >= 0.95:
                continue
            if vix.iloc[i] >= 25:
                continue

            # VPIN gate (low VPIN = noise, no informed flow)
            if pd.isna(vpin_pct[i]) or vpin_pct[i] >= self.params["vpin_pct_max"]:
                continue

            atr_i = atr[i] if i < len(atr) and not np.isnan(atr[i]) else 1.0
            proximity = self.params["gamma_wall_proximity_atr"] * atr_i

            close_i = df["close"].iloc[i]
            put_wall_i = put_wall.iloc[i]
            call_wall_i = call_wall.iloc[i]

            # LONG: near put wall + RSI2 oversold
            if (abs(close_i - put_wall_i) <= proximity
                    and not pd.isna(rsi2[i])
                    and rsi2[i] < self.params["rsi2_oversold"]):
                entry_long[i] = True
                continue

            # SHORT: near call wall + RSI2 overbought
            if (abs(close_i - call_wall_i) <= proximity
                    and not pd.isna(rsi2[i])
                    and rsi2[i] > self.params["rsi2_overbought"]):
                entry_short[i] = True

        return pd.DataFrame({
            "entry_long": entry_long, "entry_short": entry_short,
            "exit_long": exit_long, "exit_short": exit_short,
        })

    def compute_position_size(self, data: Dict[str, Any],
                              signals: pd.DataFrame) -> pd.Series:
        # Per-bar position: +1 when long, -1 when short, decay over max_hold_bars
        n = len(signals)
        pos = np.zeros(n, dtype=float)
        bars_in_pos = 0
        side = 0
        max_hold = self.params["max_hold_bars"]

        for i in range(n):
            if signals["entry_long"].iloc[i] and side == 0:
                side = 1
                bars_in_pos = 0
            elif signals["entry_short"].iloc[i] and side == 0:
                side = -1
                bars_in_pos = 0

            if side != 0:
                pos[i] = float(side)
                bars_in_pos += 1
                if bars_in_pos >= max_hold:
                    side = 0
                    bars_in_pos = 0

        return pd.Series(pos)

    def get_tunable_params(self) -> Dict[str, tuple]:
        return {
            "vrp_pct_threshold":         (60,   90,   "int"),
            "gamma_wall_proximity_atr":  (0.2,  1.0,  "float"),
            "rsi2_oversold":             (5,    25,   "int"),
            "rsi2_overbought":           (75,   95,   "int"),
            "vpin_pct_max":              (40,   60,   "int"),
            "stop_atr_mult":             (0.6,  1.6,  "float"),
            "max_hold_bars":             (7,    35,   "int"),
        }
```

- [ ] **Step 4: Run — passes**

Run: `pytest tests/test_strategy_vrp_gex.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apex/strategies/vrp_gex_fade.py tests/test_strategy_vrp_gex.py
git commit -m "phase-12b: VRPGEXFadeStrategy (Strategy 1) — gamma walls + VRP filter"
```

---

### Tasks 13-20: Strategy 1 wiring + Layer A validation

**Strategies 1-5 follow the same pattern**: data prep → signals → position sizing → registry → Optuna integration → Layer A validation. The detailed steps for each are similar; rather than repeat 500 lines per strategy, follow the pattern from Task 12 for each new strategy and apply these gating tasks at the end of each phase:

For Strategy 1, after Task 12, run:

- [ ] **Task 13:** Verify the strategy is invoked when run via the existing pipeline with `--strategy apex.strategies.vrp_gex_fade`. (Wire the registry into `apex/main.py:main()` so `--strategy` accepts a registered strategy name in addition to a file path. ~30 lines.)
- [ ] **Task 14:** Add to Layer 2 deep_tune so each registered strategy's tunable params are auto-suggested. (~20 lines.)
- [ ] **Task 15:** Run Layer A on Strategy 1 with `python apex.py --strategy vrp_gex_fade --validate-cpcv` to print 28-fold OOS Sharpe distribution. **Gate: median Sharpe > 0.3, > 55% folds positive.** If fails, document and continue to ensemble (other strategies may compensate).
- [ ] **Task 16:** Commit Layer A results doc to `docs/superpowers/results/strategy-1-vrp-gex-cpcv.md`.

(Tasks 17-20 reserved for any iteration on Strategy 1 if Layer A fails initial pass — tighten thresholds, adjust filters, etc.)

---

<!-- END_PHASE_12B -->

## Phase 12C — Strategy 2: OPEX Gravity

**Strategy 2 trades the predictable gamma-pinning around monthly OPEX expirations.** Pure calendar + options-OI structural primitive.

---

### Task 21: OPEX calendar + pin-strike helper

**Files:**
- Create: `apex/data/opex_calendar.py`
- Create: `tests/test_opex_calendar.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for OPEX calendar utilities."""
import pandas as pd


def test_third_friday_detection():
    from apex.data.opex_calendar import is_opex_week
    # Jun 2025 OPEX = Friday Jun 20, 2025
    assert is_opex_week(pd.Timestamp("2025-06-17"))  # Tuesday of OPEX week
    assert is_opex_week(pd.Timestamp("2025-06-20"))  # OPEX Friday
    assert not is_opex_week(pd.Timestamp("2025-06-10"))  # Week before OPEX
    assert not is_opex_week(pd.Timestamp("2025-06-24"))  # Week after OPEX


def test_pin_strike_lookup_picks_highest_oi(tmp_path):
    from apex.data.opex_calendar import find_pin_strike
    chain = {
        "spot": 400.0,
        "strikes": [
            {"strike": 395, "call_oi": 1000, "put_oi": 500},
            {"strike": 400, "call_oi": 5000, "put_oi": 4500},  # peak
            {"strike": 405, "call_oi": 800, "put_oi": 1200},
        ],
    }
    pin = find_pin_strike(chain, spot=400.0, window_pct=0.03)
    assert pin == 400.0


def test_pin_strike_window_excludes_far_strikes():
    from apex.data.opex_calendar import find_pin_strike
    chain = {
        "spot": 400.0,
        "strikes": [
            {"strike": 380, "call_oi": 50000, "put_oi": 50000},  # huge but outside window
            {"strike": 400, "call_oi": 1000, "put_oi": 1000},
        ],
    }
    pin = find_pin_strike(chain, spot=400.0, window_pct=0.03)  # 3% = 388-412
    assert pin == 400.0
```

- [ ] **Step 2: Run — fails**

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement apex/data/opex_calendar.py**

```python
"""OPEX calendar + pin-strike helpers for Strategy 2."""
from typing import Optional

import pandas as pd


def is_opex_week(date) -> bool:
    """Return True if `date` is in the trading week containing the third Friday."""
    ts = pd.Timestamp(date)
    # First Friday of the month
    first_day = ts.replace(day=1)
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + pd.Timedelta(days=days_until_friday)
    third_friday = first_friday + pd.Timedelta(days=14)
    # OPEX week = Mon-Fri containing third_friday
    week_start = third_friday - pd.Timedelta(days=third_friday.weekday())
    week_end = week_start + pd.Timedelta(days=4)
    return week_start.normalize() <= ts.normalize() <= week_end.normalize()


def find_pin_strike(chain: dict, spot: float,
                     window_pct: float = 0.05) -> Optional[float]:
    """Return strike with highest combined call+put OI within ±window_pct of spot.
    Returns None if no strikes in window.
    """
    if "strikes" not in chain:
        return None
    lo = spot * (1 - window_pct)
    hi = spot * (1 + window_pct)
    eligible = [s for s in chain["strikes"] if lo <= s["strike"] <= hi]
    if not eligible:
        return None
    best = max(eligible,
               key=lambda s: s.get("call_oi", 0) + s.get("put_oi", 0))
    return float(best["strike"])
```

- [ ] **Step 4: Run — passes**

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apex/data/opex_calendar.py tests/test_opex_calendar.py
git commit -m "phase-12c: OPEX calendar + pin-strike helpers"
```

---

### Task 22-32: OPEX Gravity strategy class + wiring

**Pattern: same as Strategy 1 (Tasks 11-16) but using the OPEX entry/exit logic from spec §5.2.**

For each step, the implementer creates `apex/strategies/opex_gravity.py` following:

- Class `OPEXGravityStrategy(StrategyBase)` with `name = "opex_gravity"`
- Entry on Tue/Wed of OPEX week, looking for pin strike via `find_pin_strike()`
- LONG if spot > 0.5% below pin strike; SHORT if spot > 0.5% above
- Exit on pin touch (±0.2%) or Friday close
- Stop: 1.5% beyond entry
- Tunable params per spec: `min_pin_distance_pct` (0.003-0.015), `pin_strike_window_pct` (0.03-0.08), `entry_dow` categorical, `forced_exit_dow` categorical

Tests must cover: registers in STRATEGY_REGISTRY, no entries outside OPEX week, no entries when spot is too close to pin, LONG fires below pin, SHORT fires above, exit on pin touch, forced Friday exit.

**Gating: same as Phase 12B Tasks 13-16.** Run Layer A CPCV. Gate: median Sharpe > 0.3, > 55% folds positive. Document results.

Concrete commit per task: `phase-12c: <action>`.

---

<!-- END_PHASE_12C -->

## Phase 12D — Strategy 3: VIX Term Structure Trade

**Strategy 3 trades the VIX/VIX3M ratio mean-reversion (curve, not level).**

---

### Task 33-42: VIX term structure strategy

**Files:**
- Create: `apex/strategies/vix_term_structure.py`
- Create: `tests/test_strategy_vix_term.py`

**Pattern: same as Strategy 1.** Implementation per spec §5.3:

- Class `VIXTermStructureStrategy(StrategyBase)` with `name = "vix_term_structure"`
- Compute `ts_ratio = VIX / VIX3M` daily; forward-fill onto 1H exec bars
- Compute 5-day RSI on `ts_ratio` itself for confirmation
- LONG SPY when `ts_ratio < contango_extreme_threshold` (default 0.85) AND ts_ratio_RSI < 30
- SHORT SPY when `ts_ratio > backwardation_extreme_threshold` (default 1.10) AND ts_ratio_RSI > 70
- Exit on revert to neutral band (0.95-1.02) OR time stop (10 bars) OR ATR-1.5 stop
- Tunable params per spec §5.3 (6 params)

Tests: registers, no entries when ratio in neutral band, LONG fires on extreme contango + RSI confirm, SHORT fires on extreme backwardation, exit on neutral revert, exit on time stop.

**Gate:** Layer A CPCV. Median Sharpe > 0.3.

Commit prefix: `phase-12d:`.

---

<!-- END_PHASE_12D -->

## Phase 12E — Strategy 4: Volatility Skew Arbitrage

**Strategy 4 trades 25-delta put/call IV ratio extremes from the Polygon options surface.** Most data-intensive of the 5 — needs full chain processing.

---

### Task 43-44: Skew computation helper

**Files:**
- Create: `apex/data/vol_skew.py`
- Create: `tests/test_vol_skew.py`

```python
def compute_skew_ratio(chain: dict, dte_target: int = 30,
                       delta_target: float = 0.25) -> Optional[float]:
    """Return IV(25-delta put) / IV(25-delta call) for the chain at given DTE.
    Selects nearest expiry to dte_target, then nearest contracts to ±delta_target.
    Returns None if either side missing.
    """
```

Tests: known-symmetric chain → ratio ≈ 1.0; put-skewed chain → ratio > 1.0; missing data → None.

---

### Task 45-56: Vol skew arbitrage strategy

**Files:**
- Create: `apex/strategies/vol_skew_arb.py`
- Create: `tests/test_strategy_vol_skew.py`

**Pattern same.** Per spec §5.4:

- Class `VolSkewArbStrategy(StrategyBase)` with `name = "vol_skew_arb"`
- Compute daily skew ratio from Polygon options chain via `compute_skew_ratio`
- Forward-fill skew ratio onto 1H exec bars
- LONG SPY when `skew > put_skew_extreme` (default 1.30); SHORT when `skew < call_skew_extreme` (default 0.95)
- Exit on revert to normal band (1.05-1.20) OR time stop (5 days = ~35 bars) OR ATR-1.0 stop
- Tunable params per spec §5.4 (7 params)

Tests cover: registers, no entries in normal band, extreme put skew → long, extreme call skew → short, exit on revert, time-stop exit.

**Gate:** Layer A CPCV. Median Sharpe > 0.3.

Commit prefix: `phase-12e:`.

---

<!-- END_PHASE_12E -->

## Phase 12F — Strategy 5: SMC Structural

**Strategy 5: pure price-structure entries (FVG + order blocks). No indicators.** Extends Phase 2 FVG detector.

---

### Task 57-58: Order block detector

**Files:**
- Create: `apex/indicators/order_blocks.py`
- Create: `tests/test_order_blocks.py`

```python
def detect_order_blocks(df, min_body_ratio=0.5):
    """Return list of order blocks. An order block is a 3-bar pattern where:
      - bar[i]: down close (red)
      - bar[i+1]: inside or small body
      - bar[i+2]: strong up close > bar[i].open AND body_ratio > min_body_ratio
    Mirror for bearish OBs.
    Each OB record: {start_idx, direction, low, high, mitigated_at_idx}.
    """
```

Tests: bullish OB detected on red→inside→strong-green, bearish OB mirrored, mitigation tracking (price returns into OB zone), no false positives on continuous trend.

---

### Task 59-68: SMC structural strategy

**Files:**
- Create: `apex/strategies/smc_structural.py`
- Create: `tests/test_strategy_smc.py`

**Pattern same.** Per spec §5.5:

- Class `SMCStructuralStrategy(StrategyBase)` with `name = "smc_structural"`
- Use existing `detect_fvgs` (Phase 2) + new `detect_order_blocks`
- LONG: price retest of unfilled bullish FVG OR bullish order block + VIX < 25 + VPIN < 50pct
- SHORT: mirror
- Exit: FVG fully filled OR opposite-direction FVG forms OR 16-bar time stop OR dynamic FVG trail (Phase 6 stops)
- Tunable params per spec §5.5 (5 params)

Tests cover: registers, no entries when VIX too high, no entries when VPIN too high, LONG fires on bullish FVG retest with confluence, SHORT mirror, FVG-fill exit, opposite-FVG exit.

**Gate:** Layer A CPCV. Median Sharpe > 0.3.

Commit prefix: `phase-12f:`.

---

<!-- END_PHASE_12F -->

## Phase 12G — Strategy 6: Cross-Asset Vol Regime Overlay

**Strategy 6 is an OVERLAY (multiplies position sizing of strategies 1-5), not a standalone signal.**

---

### Task 69: MOVE + OVX fetchers

**Files:**
- Create: `apex/data/cross_asset_vol.py`
- Create: `tests/test_cross_asset_vol.py`

```python
def fetch_move_index(start, end, cache_dir):
    """Fetch ICE BofA MOVE Index (rates vol) — FRED series 'BAMLH0A0HYM2EY' as proxy."""

def fetch_ovx(start, end, cache_dir):
    """Fetch CBOE Crude Oil Volatility Index — FRED 'OVXCLS'."""

def compute_vol_percentiles(vix, move, ovx, window=252):
    """Return DataFrame with vix_pct, move_pct, ovx_pct columns."""
```

Tests: caching works, percentile in [0, 100], NaN handling, all three series merge correctly on date.

---

### Task 70-71: Cross-asset overlay class

**Files:**
- Create: `apex/strategies/cross_asset_vol_overlay.py`
- Create: `tests/test_strategy_overlay.py`

```python
@register_strategy
class CrossAssetVolOverlayStrategy(StrategyBase):
    name = "cross_asset_vol_overlay"
    data_requirements = ["vix_pct", "move_pct", "ovx_pct"]

    def compute_signals(self, data):
        # Overlay never emits trade signals — it only sizes others.
        n = len(data["exec_df_1H"])
        return pd.DataFrame({c: [False]*n for c in
                             ("entry_long","entry_short","exit_long","exit_short")})

    def compute_position_size(self, data, signals):
        """Returns the SIZE MULTIPLIER (not a directional signal).
        Combiner detects this overlay by name and applies it as a scalar."""
        ...
```

Logic per spec §5.6:
- All three pcts > 80 → 0.5 (risk-off, scale down)
- All three pcts < 20 → 1.2 (risk-on, scale up)
- Divergent (e.g., VIX high but MOVE low) → 1.0
- Default → 1.0

Tests: all-high → 0.5, all-low → 1.2, divergent → 1.0, missing data → 1.0 (graceful).

---

### Task 72-74: Combiner integration of overlay

**Modify:** `apex/ensemble/combiner.py`

Detect when one of the strategies is named `"cross_asset_vol_overlay"`. Treat its `compute_position_size()` output as a per-bar **multiplier** applied to the combined portfolio position, NOT as a signal to sum.

```python
# Inside EnsembleCombiner.run():
overlay_mult = pd.Series([1.0] * n)
non_overlay_strategies = []
for s in self.strategies:
    if s.name == "cross_asset_vol_overlay":
        overlay_mult = s.compute_position_size(data, s.compute_signals(data))
    else:
        non_overlay_strategies.append(s)
# ... combine non_overlay_strategies as before, then:
combined = combined * overlay_mult.values
```

Tests: combiner passes overlay through; final position is scaled appropriately.

Commit prefix: `phase-12g:`.

---

<!-- END_PHASE_12G -->

## Phase 12H — Ensemble Integration + Layers B/C Validation

---

### Task 75: Wire ensemble into main pipeline

**Modify:** `apex/main.py`

Add `--ensemble` CLI flag. When set:
1. Load all registered strategies from STRATEGY_REGISTRY (or a curated list from config)
2. For each strategy: run Layer 2 deep_tune to find best params
3. Build EnsembleCombiner with tuned strategies
4. Run combiner on tune_df, then on holdout_df
5. Compute ensemble portfolio NAV
6. Pass to ensemble CPCV + walk-forward weight validation
7. Generate ensemble HTML report

Config additions:
```jsonc
"ensemble": {
    "enabled": false,
    "strategies": ["vrp_gex_fade", "opex_gravity", "vix_term_structure",
                   "vol_skew_arb", "smc_structural", "cross_asset_vol_overlay"],
    "max_weight": 0.30,
    "vol_lookback_days": 60,
    "size_change_threshold": 0.10
}
```

Tests: `--ensemble` flag dispatch works; legacy single-strategy mode unaffected; golden snapshot still byte-equal.

Commit: `phase-12h: --ensemble CLI flag + dispatch`.

---

### Task 76: Per-strategy data preparation

**Modify:** `apex/main.py`

Each strategy declares `data_requirements`. Main pipeline must satisfy them:
- `exec_df_1H` ✓ (already fetched in phase3_fetch_data)
- `regime_state` ✓ (already merged via `compute_vrp_regime`)
- `vix`, `vxv`, `vrp_pct` ✓ (Phase 1 merge)
- `call_wall`, `put_wall`, `gamma_flip` ← needs `ingest_flux_points` enabled
- `move_pct`, `ovx_pct` ← NEW: fetch via `apex.data.cross_asset_vol`

Add a `prepare_ensemble_data(data_dict, cfg)` function that augments each symbol's `exec_df` with all required columns. Strategy missing a column gracefully degrades to no-trade for that bar.

Tests: prepared data has all required columns; strategies still work when one requirement is missing.

Commit: `phase-12h: prepare_ensemble_data adds all required columns per strategy`.

---

### Task 77-80: Run Layer A on each strategy

For each strategy that survives the import:
- Run individual Optuna Layer 2 (150 trials/symbol) → best_params per symbol
- Run CPCV on the tuned best_params (28 folds)
- If median Sharpe > 0.3 AND >55% folds positive → mark as Layer A pass
- If fails → exclude from ensemble

Output `apex_results/<run>/strategy_layer_a_results.csv` with one row per (strategy, symbol).

Commit per task: `phase-12h: Layer A validation for strategy <N>`.

---

### Task 81-82: Run Layer B (ensemble CPCV)

After all Layer A passes are collected:
1. Build EnsembleCombiner with surviving strategies
2. Run combiner on tune window → portfolio_returns_tune
3. Pass portfolio_returns_tune to `evaluate_ensemble_cpcv` → 28-fold distribution
4. **Layer B Gate:** ensemble median Sharpe > 0.8 AND >65% folds positive

Output `apex_results/<run>/ensemble_layer_b_results.json`.

Commit: `phase-12h: Layer B ensemble CPCV validation`.

---

### Task 83-84: Run Layer C (walk-forward weights)

1. Compute monthly returns per strategy across the full backtest window
2. Run `compare_dynamic_vs_static_weights` to validate that monthly weight refresh actually adds Sharpe
3. **Layer C Gate:** dynamic Sharpe ≥ static Sharpe + 0.05

If Layer C fails → log warning but don't reject (static weights might be fine in low-regime-shift periods).

Commit: `phase-12h: Layer C walk-forward weight validation`.

---

### Task 85-86: End-to-end ensemble run

- [ ] Run `python apex.py --ensemble --budget medium --no-amibroker`. Verify:
  1. Each strategy individually tunes (Layer A reports)
  2. Ensemble combiner produces portfolio NAV
  3. Layer B reports ensemble Sharpe distribution
  4. Layer C reports weight uplift
  5. HTML report generated with ensemble summary
- [ ] Commit any final fixes: `phase-12h: end-to-end ensemble run validation`.

---

<!-- END_PHASE_12H -->

## Phase 12I — HTML Report + CLI

---

### Task 87-90: Ensemble HTML report

**Files:**
- Create: `apex/report/ensemble_report.py`

Layout:
- Tab 1: Headline numbers — ensemble Sharpe (CPCV median + IQR), max DD, total return
- Tab 2: Per-strategy contributions — weight, Sharpe, return contribution
- Tab 3: Equity curve overlay — combined vs each strategy
- Tab 4: Regime breakdown — performance per regime (R1/R2/R3)
- Tab 5: CPCV distribution — histogram of 28-fold OOS Sharpes
- Tab 6: Walk-forward weights — dynamic vs static comparison
- Tab 7: Layer A results — per-strategy CPCV outcomes

Tests: HTML file written without errors, all tab content present, JSON validity in embedded data.

Commit: `phase-12i: ensemble HTML report`.

---

### Task 91-92: Final integration + push

- [ ] Run end-to-end with default config and CPCV enabled
- [ ] Verify ensemble report opens cleanly in browser
- [ ] Update README with `--ensemble` usage
- [ ] Update CHANGELOG with v3.0 ensemble release notes
- [ ] Commit final state, push to GitHub

Commits:
- `phase-12i: README + CHANGELOG for ensemble release`
- `phase-12i: final ensemble integration verified`

---

<!-- END_OF_PLAN -->

