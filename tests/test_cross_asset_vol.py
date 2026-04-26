"""Tests for cross-asset vol helpers (MOVE + OVX + percentiles)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apex.data import cross_asset_vol


def test_compute_vol_percentiles_in_range():
    """Percentile values should always lie in [0, 100] (or be NaN at warmup)."""
    rng = np.random.default_rng(17)
    n = 600
    vix = pd.Series(rng.uniform(10, 30, n))
    move = pd.Series(rng.uniform(50, 120, n))
    ovx = pd.Series(rng.uniform(20, 60, n))
    out = cross_asset_vol.compute_vol_percentiles(vix, move, ovx, window=252)
    for col in ("vix_pct", "move_pct", "ovx_pct"):
        assert col in out.columns
        non_nan = out[col].dropna()
        assert (non_nan >= 0).all()
        assert (non_nan <= 100).all()


def test_percentiles_handle_nan_input():
    """NaN inputs should propagate to NaN percentiles, not crash."""
    n = 300
    vix = pd.Series([np.nan] * 50 + [15.0] * (n - 50))
    move = pd.Series([np.nan] * 50 + [80.0] * (n - 50))
    ovx = pd.Series([np.nan] * 50 + [35.0] * (n - 50))
    out = cross_asset_vol.compute_vol_percentiles(vix, move, ovx, window=252)
    assert out["vix_pct"].isna().any()
    # After enough non-NaN data, percentiles should compute (warmup = 252//2 = 126)
    assert out["vix_pct"].notna().any()


def test_fetch_move_index_uses_cache(monkeypatch, tmp_path):
    """fetch_move_index must call fetch_fred_series with the BAMLH0A0HYM2EY id."""
    captured = {}

    def fake_fetch(series_id, start, end, cache_dir=None):
        captured["series_id"] = series_id
        captured["cache_dir"] = cache_dir
        return pd.DataFrame({"value": [1.0, 2.0]},
                            index=pd.to_datetime(["2025-01-01", "2025-01-02"]))

    monkeypatch.setattr(cross_asset_vol, "fetch_fred_series", fake_fetch)
    df = cross_asset_vol.fetch_move_index("2025-01-01", "2025-01-31",
                                            cache_dir=tmp_path)
    assert captured["series_id"] == "BAMLH0A0HYM2EY"
    assert captured["cache_dir"] == tmp_path
    assert len(df) == 2


def test_fetch_ovx_uses_correct_series(monkeypatch, tmp_path):
    captured = {}

    def fake_fetch(series_id, start, end, cache_dir=None):
        captured["series_id"] = series_id
        return pd.DataFrame({"value": [10.0]}, index=pd.to_datetime(["2025-01-01"]))

    monkeypatch.setattr(cross_asset_vol, "fetch_fred_series", fake_fetch)
    df = cross_asset_vol.fetch_ovx("2025-01-01", "2025-01-31", cache_dir=tmp_path)
    assert captured["series_id"] == "OVXCLS"
    assert len(df) == 1
