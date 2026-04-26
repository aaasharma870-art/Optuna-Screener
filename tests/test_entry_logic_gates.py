"""Verify each VRP entry gate fires/blocks correctly per the doc spec.

Each test fabricates a minimal `signals_data` dict with controlled
`extras` values and exercises `determine_entry_direction` directly.

These tests cover the seven bugs fixed in Phase 8:
    1. R3 VPIN gate was inverted (now requires HIGH VPIN).
    2. R3 VPIN read raw vpin instead of percentile.
    3. R1/R2 fades missing low-VPIN gate.
    4. R3 used raw vwap_slope instead of ATR-normalized.
    5. R1/R2 missing breakout-reversal gate.
    6. R3 missing sweep-proxy gate.
    7. cum_vwclv default threshold bumped from 1.0 to 1.3.
"""
import math
import pandas as pd
import pytest


def _make_signals_data(n=20, **overrides):
    """Build a minimal signals_data with extras for entry-logic testing."""
    base = {
        "rsi2":              pd.Series([50.0] * n),
        "cum_vwclv":         pd.Series([0.0] * n),
        "vwap_slope":        pd.Series([0.0] * n),
        "vwap_slope_atr":    pd.Series([0.0] * n),
        "vpin":              pd.Series([0.5] * n),
        "vpin_pct":          pd.Series([50.0] * n),
        "vwap_lower":        pd.Series([100.0] * n),
        "vwap_upper":        pd.Series([100.0] * n),
        "close":             pd.Series([100.0] * n),
        "vrp_pct":           pd.Series([50.0] * n),
        "vix":               pd.Series([15.0] * n),
        "breakout_reversal_long":  pd.Series([False] * n),
        "breakout_reversal_short": pd.Series([False] * n),
        "sweep_proxy_long":        pd.Series([False] * n),
        "sweep_proxy_short":       pd.Series([False] * n),
    }
    base.update(overrides)
    return {
        "signals": {},
        "regime": pd.Series(["R1"] * n),
        "score": pd.Series([0] * n),
        "atr": pd.Series([1.0] * n),
        "extras": base,
    }


def test_r3_long_requires_high_vpin():
    """R3 long entry must require vpin_pct > 60 (HIGH VPIN, informed flow)."""
    from apex.engine.backtest import determine_entry_direction
    n = 20
    # All R3 long conditions met EXCEPT VPIN is low (40)
    sd = _make_signals_data(
        n,
        cum_vwclv=pd.Series([2.0] * n),
        vwap_slope_atr=pd.Series([0.5] * n),
        vpin_pct=pd.Series([40.0] * n),  # too low for R3
        sweep_proxy_long=pd.Series([True] * n),
    )
    direction, _ = determine_entry_direction("R3", 0, sd, 5, {})
    assert direction is None, "R3 should NOT fire long with vpin_pct=40 (too low)"
    # Now bump to 70 (above threshold) -- should fire
    sd2 = _make_signals_data(
        n,
        cum_vwclv=pd.Series([2.0] * n),
        vwap_slope_atr=pd.Series([0.5] * n),
        vpin_pct=pd.Series([70.0] * n),  # above 60 threshold
        sweep_proxy_long=pd.Series([True] * n),
    )
    direction, _ = determine_entry_direction("R3", 0, sd2, 5, {})
    assert direction == "long", "R3 SHOULD fire long with vpin_pct=70 + all other conds met"


def test_r1_long_requires_low_vpin():
    """R1 long fade must require vpin_pct < 50 (LOW VPIN, noise environment)."""
    from apex.engine.backtest import determine_entry_direction
    n = 20
    # All R1 long conditions met EXCEPT VPIN is high (70)
    sd = _make_signals_data(
        n,
        rsi2=pd.Series([10.0] * n),
        cum_vwclv=pd.Series([0.5] * n),
        vpin_pct=pd.Series([70.0] * n),  # too high for R1 fade
        breakout_reversal_long=pd.Series([True] * n),
    )
    direction, _ = determine_entry_direction("R1", 0, sd, 5, {})
    assert direction is None, "R1 should NOT fire fade with vpin_pct=70 (too high)"
    # Drop to 30 -- should fire
    sd2 = _make_signals_data(
        n,
        rsi2=pd.Series([10.0] * n),
        cum_vwclv=pd.Series([0.5] * n),
        vpin_pct=pd.Series([30.0] * n),  # below 50 threshold
        breakout_reversal_long=pd.Series([True] * n),
    )
    direction, _ = determine_entry_direction("R1", 0, sd2, 5, {})
    assert direction == "long", "R1 SHOULD fire long fade with vpin_pct=30 + all other conds"


def test_r1_requires_breakout_reversal_long():
    """R1 long fade must require breakout_reversal_long signal."""
    from apex.engine.backtest import determine_entry_direction
    n = 20
    sd = _make_signals_data(
        n,
        rsi2=pd.Series([10.0] * n),
        cum_vwclv=pd.Series([0.5] * n),
        vpin_pct=pd.Series([30.0] * n),
        breakout_reversal_long=pd.Series([False] * n),  # gate blocks
    )
    direction, _ = determine_entry_direction("R1", 0, sd, 5, {})
    assert direction is None, "R1 must require breakout_reversal_long=True"


def test_short_whitelist_blocks_non_listed_symbol_regime():
    """Phase 9: shorts gated by (symbol, regime) whitelist.

    Diagnostics on run #4 showed shorts only have edge on (SPY, R1).
    Whitelist enforces this — shorts on (QQQ, R1) must be blocked.
    """
    from apex.engine.backtest import determine_entry_direction
    n = 20
    # All R1 short conditions met
    sd = _make_signals_data(
        n,
        rsi2=pd.Series([90.0] * n),
        cum_vwclv=pd.Series([-0.5] * n),
        vpin_pct=pd.Series([30.0] * n),
        breakout_reversal_short=pd.Series([True] * n),
    )
    # Whitelist allows (SPY, R1) only — QQQ R1 short must be blocked
    params = {"symbol": "QQQ", "vrp_short_whitelist": [["SPY", "R1"]]}
    direction, _ = determine_entry_direction("R1", 0, sd, 5, params)
    assert direction is None, "QQQ R1 short blocked by whitelist"

    # SPY R1 short with same conditions should fire
    params2 = {"symbol": "SPY", "vrp_short_whitelist": [["SPY", "R1"]]}
    direction2, _ = determine_entry_direction("R1", 0, sd, 5, params2)
    assert direction2 == "short", "SPY R1 short allowed by whitelist"


def test_short_whitelist_empty_allows_all():
    """Empty/missing whitelist preserves back-compat (allows all shorts)."""
    from apex.engine.backtest import determine_entry_direction
    n = 20
    sd = _make_signals_data(
        n,
        rsi2=pd.Series([90.0] * n),
        cum_vwclv=pd.Series([-0.5] * n),
        vpin_pct=pd.Series([30.0] * n),
        breakout_reversal_short=pd.Series([True] * n),
    )
    # No whitelist — any symbol/regime can short
    params = {"symbol": "QQQ"}
    direction, _ = determine_entry_direction("R1", 0, sd, 5, params)
    assert direction == "short", "Without whitelist, shorts allowed for any symbol"


def test_r3_does_not_require_sweep_proxy():
    """Phase 9: R3 sweep_proxy is bonus-only (was required in Phase 8).

    The 4-AND from Phase 8 made R3 impossible to satisfy at 1H granularity
    (zero R3 trades fired in run #4). Phase 9 dropped sweep_proxy from the
    required gate set so R3 trends can actually trigger.
    """
    from apex.engine.backtest import determine_entry_direction
    n = 20
    sd = _make_signals_data(
        n,
        cum_vwclv=pd.Series([2.0] * n),
        vwap_slope_atr=pd.Series([0.5] * n),
        vpin_pct=pd.Series([70.0] * n),
        sweep_proxy_long=pd.Series([False] * n),  # no longer blocks
    )
    direction, _ = determine_entry_direction("R3", 0, sd, 5, {})
    assert direction == "long", "R3 should fire long without sweep_proxy after Phase 9"


def test_r3_uses_atr_normalized_slope():
    """R3 must use vwap_slope_atr (ATR-normalized) > 0.2, not raw slope."""
    from apex.engine.backtest import determine_entry_direction
    n = 20
    # Raw slope positive but ATR-normalized below threshold
    sd = _make_signals_data(
        n,
        cum_vwclv=pd.Series([2.0] * n),
        vwap_slope=pd.Series([0.001] * n),     # tiny raw slope
        vwap_slope_atr=pd.Series([0.05] * n),  # below 0.2 threshold
        vpin_pct=pd.Series([70.0] * n),
        sweep_proxy_long=pd.Series([True] * n),
    )
    direction, _ = determine_entry_direction("R3", 0, sd, 5, {})
    assert direction is None, "R3 must require vwap_slope_atr > 0.2"


def test_r4_no_trade():
    from apex.engine.backtest import determine_entry_direction
    direction, size = determine_entry_direction(
        "R4", 0, _make_signals_data(20), 5, {},
    )
    assert direction is None
    assert size == 0.0


def test_cum_vwclv_threshold_default_1_3():
    """Default cum_vwclv threshold should be 1.3 per doc, not 1.0."""
    from apex.engine.backtest import determine_entry_direction
    n = 20
    # cum_vwclv = 1.1 (above old default 1.0, below new default 1.3)
    sd = _make_signals_data(
        n,
        cum_vwclv=pd.Series([1.1] * n),
        vwap_slope_atr=pd.Series([0.5] * n),
        vpin_pct=pd.Series([70.0] * n),
        sweep_proxy_long=pd.Series([True] * n),
    )
    direction, _ = determine_entry_direction("R3", 0, sd, 5, {})
    assert direction is None, (
        "cum_vwclv=1.1 should be below new default threshold 1.3"
    )
