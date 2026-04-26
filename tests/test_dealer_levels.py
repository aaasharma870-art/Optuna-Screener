"""Tests for apex.data.dealer_levels.ingest_flux_points."""
from pathlib import Path

import pandas as pd
import pytest


def test_dealer_levels_adds_columns(monkeypatch, tmp_path):
    from apex.data import dealer_levels

    fix = Path(__file__).parent / "fixtures" / "SPY_1H.parquet"
    edf = pd.read_parquet(fix).head(50).copy()

    fake_levels = {
        "call_wall": 450.0, "put_wall": 430.0,
        "gamma_flip": 440.0, "vol_trigger": 374.0,
        "abs_gamma_strike": 445.0,
    }

    def _stub_compute_gex_proxy(symbol, as_of, cache_dir):
        return dict(fake_levels)

    monkeypatch.setattr("apex.data.dealer_levels.compute_gex_proxy",
                        _stub_compute_gex_proxy)
    out = dealer_levels.ingest_flux_points(edf, "SPY", tmp_path)
    for col in ("call_wall", "put_wall", "gamma_flip",
                "vol_trigger", "abs_gamma_strike"):
        assert col in out.columns
    # All bars on the same day should get the same call_wall value
    assert (out["call_wall"] == 450.0).all()


def test_dealer_levels_handles_compute_failure(monkeypatch, tmp_path):
    from apex.data import dealer_levels
    fix = Path(__file__).parent / "fixtures" / "SPY_1H.parquet"
    edf = pd.read_parquet(fix).head(20).copy()

    def _raise(*_a, **_k):
        raise RuntimeError("simulated chain fetch failure")

    monkeypatch.setattr("apex.data.dealer_levels.compute_gex_proxy", _raise)
    out = dealer_levels.ingest_flux_points(edf, "SPY", tmp_path)
    # Columns must exist but be all-NaN
    assert "call_wall" in out.columns
    assert out["call_wall"].isna().all()


def test_dealer_levels_no_datetime_returns_unchanged(tmp_path):
    from apex.data import dealer_levels
    edf = pd.DataFrame({"close": [100.0, 101.0]})
    out = dealer_levels.ingest_flux_points(edf, "SPY", tmp_path)
    # No datetime column -> no augmentation
    assert "call_wall" not in out.columns
