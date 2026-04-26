"""Tests for Strategy 6: Cross-Asset Vol Overlay."""
import numpy as np
import pandas as pd


def _make_data(n=50, vix_pct=50.0, move_pct=50.0, ovx_pct=50.0):
    """Build data with constant percentile columns by default."""
    rng = np.random.default_rng(19)
    close = 400 + np.cumsum(rng.normal(0, 0.5, n))
    return {
        "exec_df_1H": pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:30", periods=n, freq="h"),
            "open": close - 0.05, "high": close + 0.20,
            "low": close - 0.20, "close": close,
            "volume": rng.integers(10000, 50000, n).astype(float),
            "vix_pct": np.full(n, float(vix_pct)),
            "move_pct": np.full(n, float(move_pct)),
            "ovx_pct": np.full(n, float(ovx_pct)),
        }),
        "symbol": "SPY",
    }


def test_overlay_registers():
    from apex.strategies import STRATEGY_REGISTRY
    from apex.strategies import cross_asset_vol_overlay  # noqa: F401
    assert "cross_asset_vol_overlay" in STRATEGY_REGISTRY


def test_no_tunable_params():
    from apex.strategies.cross_asset_vol_overlay import CrossAssetVolOverlayStrategy
    s = CrossAssetVolOverlayStrategy()
    assert s.get_tunable_params() == {}


def test_signals_are_all_false():
    """Overlay never emits trade signals — every signal column is all-False."""
    from apex.strategies.cross_asset_vol_overlay import CrossAssetVolOverlayStrategy
    s = CrossAssetVolOverlayStrategy()
    data = _make_data(n=30, vix_pct=85, move_pct=85, ovx_pct=85)
    signals = s.compute_signals(data)
    for c in ("entry_long", "entry_short", "exit_long", "exit_short"):
        assert not signals[c].any()


def test_all_high_returns_05():
    """All three pcts > 80 -> 0.5 size multiplier (risk-off)."""
    from apex.strategies.cross_asset_vol_overlay import CrossAssetVolOverlayStrategy
    s = CrossAssetVolOverlayStrategy()
    data = _make_data(n=30, vix_pct=85, move_pct=90, ovx_pct=82)
    sig = s.compute_signals(data)
    mult = s.compute_position_size(data, sig)
    assert (mult == 0.5).all()


def test_all_low_returns_12():
    """All three pcts < 20 -> 1.2 size multiplier (risk-on)."""
    from apex.strategies.cross_asset_vol_overlay import CrossAssetVolOverlayStrategy
    s = CrossAssetVolOverlayStrategy()
    data = _make_data(n=30, vix_pct=10, move_pct=15, ovx_pct=18)
    sig = s.compute_signals(data)
    mult = s.compute_position_size(data, sig)
    assert (mult == 1.2).all()


def test_divergent_returns_10():
    """Divergent (e.g., VIX high but MOVE/OVX low) -> 1.0 (neutral)."""
    from apex.strategies.cross_asset_vol_overlay import CrossAssetVolOverlayStrategy
    s = CrossAssetVolOverlayStrategy()
    data = _make_data(n=30, vix_pct=85, move_pct=15, ovx_pct=50)
    sig = s.compute_signals(data)
    mult = s.compute_position_size(data, sig)
    assert (mult == 1.0).all()


def test_missing_data_returns_10():
    """If percentile columns absent or NaN -> default 1.0 (neutral)."""
    from apex.strategies.cross_asset_vol_overlay import CrossAssetVolOverlayStrategy
    s = CrossAssetVolOverlayStrategy()
    data = _make_data(n=30)
    # Drop the percentile columns
    data["exec_df_1H"] = data["exec_df_1H"].drop(
        columns=["vix_pct", "move_pct", "ovx_pct"])
    sig = s.compute_signals(data)
    mult = s.compute_position_size(data, sig)
    assert (mult == 1.0).all()
