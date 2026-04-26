"""Tests for Strategy 2: OPEX Gravity."""
import numpy as np
import pandas as pd


def _make_test_data(n=200, start="2026-04-13 09:30", spot=400.0,
                     pin_strike=400.0):
    """Build minimal data with OPEX-week dates and a pre-merged pin_strike column."""
    rng = np.random.default_rng(7)
    close = spot + np.cumsum(rng.normal(0, 0.10, n)) * 0.0  # flat by default
    return {
        "exec_df_1H": pd.DataFrame({
            "datetime": pd.date_range(start, periods=n, freq="h"),
            "open": close - 0.05, "high": close + 0.20,
            "low": close - 0.20, "close": close,
            "volume": rng.integers(10000, 50000, n).astype(float),
            "pin_strike": np.full(n, pin_strike),
        }),
        "symbol": "SPY",
    }


def test_opex_gravity_registers():
    from apex.strategies import STRATEGY_REGISTRY
    from apex.strategies import opex_gravity  # noqa: F401
    assert "opex_gravity" in STRATEGY_REGISTRY


def test_no_entries_outside_opex_week():
    """A non-OPEX week (e.g., Apr 6-10, 2026) must produce zero entries."""
    from apex.strategies.opex_gravity import OPEXGravityStrategy
    s = OPEXGravityStrategy()
    # April 6, 2026 is a Monday — not OPEX week (3rd Friday is Apr 17).
    data = _make_test_data(n=80, start="2026-04-06 09:30",
                            spot=395.0, pin_strike=400.0)
    signals = s.compute_signals(data)
    assert not signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_no_entries_when_close_to_pin():
    """If price is within min_pin_distance_pct of pin, no signal fires."""
    from apex.strategies.opex_gravity import OPEXGravityStrategy
    s = OPEXGravityStrategy()
    # Spot exactly equals pin -> distance = 0 -> no entry
    data = _make_test_data(n=80, start="2026-04-13 09:30",
                            spot=400.0, pin_strike=400.0)
    signals = s.compute_signals(data)
    assert not signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_long_fires_below_pin():
    """Spot well below pin during OPEX-week Tue/Wed -> LONG entry."""
    from apex.strategies.opex_gravity import OPEXGravityStrategy
    s = OPEXGravityStrategy()
    # 2026-04-13 (Mon) start, default Tue-Wed entry window. Spot=395, pin=400 -> +1.27% above spot.
    data = _make_test_data(n=80, start="2026-04-13 09:30",
                            spot=395.0, pin_strike=400.0)
    signals = s.compute_signals(data)
    assert signals["entry_long"].any()
    assert not signals["entry_short"].any()


def test_short_fires_above_pin():
    """Spot well above pin during OPEX-week Tue/Wed -> SHORT entry."""
    from apex.strategies.opex_gravity import OPEXGravityStrategy
    s = OPEXGravityStrategy()
    data = _make_test_data(n=80, start="2026-04-13 09:30",
                            spot=405.0, pin_strike=400.0)
    signals = s.compute_signals(data)
    assert signals["entry_short"].any()
    assert not signals["entry_long"].any()


def test_forced_friday_exit():
    """Position must be zeroed by Friday afternoon close."""
    from apex.strategies.opex_gravity import OPEXGravityStrategy
    s = OPEXGravityStrategy()
    # Long signal in OPEX week
    data = _make_test_data(n=120, start="2026-04-14 09:30",
                            spot=395.0, pin_strike=400.0)  # Tuesday start
    signals = s.compute_signals(data)
    pos = s.compute_position_size(data, signals)
    # After Friday afternoon, position must be flat (exit AT Friday close
    # is allowed — the next bar onward must be zero).
    dt = pd.to_datetime(data["exec_df_1H"]["datetime"])
    friday_pm_mask = (dt.dt.weekday == 4) & (dt.dt.hour >= 12)
    if friday_pm_mask.any():
        first_fri_pm_idx = int(friday_pm_mask.values.argmax())
        # All bars STRICTLY after the exit bar must be flat
        assert (pos.iloc[first_fri_pm_idx + 1:] == 0).all()


def test_tunable_params_match_spec():
    from apex.strategies.opex_gravity import OPEXGravityStrategy
    s = OPEXGravityStrategy()
    params = s.get_tunable_params()
    expected = {"min_pin_distance_pct", "pin_strike_window_pct",
                "entry_dow", "forced_exit_dow"}
    assert set(params.keys()) == expected
