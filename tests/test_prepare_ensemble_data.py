"""Tests for prepare_ensemble_data data-augmentation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _fixture_data_dict():
    fix = Path(__file__).parent / "fixtures" / "SPY_1H.parquet"
    edf = pd.read_parquet(fix)
    return {
        "SPY": {
            "exec_df": edf.copy(),
            "exec_df_holdout": edf.iloc[:50].copy(),
            "daily_df": None,
            "daily_df_holdout": None,
        }
    }


def test_prepare_ensemble_data_runs_without_crash(monkeypatch):
    """When external data sources are unavailable, the helper should
    log warnings and return the data_dict unchanged (or partially augmented),
    without raising."""
    from apex import main_ensemble

    # Stub FRED/MOVE/OVX so we don't hit the network in CI
    def _empty_df(*_a, **_k):
        return pd.DataFrame(columns=["value"])

    monkeypatch.setattr("apex.data.cross_asset_vol.fetch_move_index", _empty_df)
    monkeypatch.setattr("apex.data.cross_asset_vol.fetch_ovx", _empty_df)
    monkeypatch.setattr("apex.data.fred_client.fetch_fred_series", _empty_df)

    cfg = {"cache_dir": "apex_cache", "options_gex": {"enabled": False}}
    out = main_ensemble.prepare_ensemble_data(_fixture_data_dict(), cfg)
    assert "SPY" in out
    edf = out["SPY"]["exec_df"]
    assert isinstance(edf, pd.DataFrame)
    assert len(edf) > 0


def test_prepare_ensemble_data_adds_vol_pct_columns(monkeypatch):
    """When FRED returns data, vix_pct/move_pct/ovx_pct columns should be merged."""
    from apex import main_ensemble

    fix = Path(__file__).parent / "fixtures" / "SPY_1H.parquet"
    edf = pd.read_parquet(fix)
    sym_dt = pd.to_datetime(edf["datetime"])
    start = sym_dt.min() - pd.Timedelta(days=400)
    end = sym_dt.max() + pd.Timedelta(days=2)

    # Synthesize a daily VIX/MOVE/OVX series spanning the bar range
    daily = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(42)
    fake_vix = pd.DataFrame({"value": rng.uniform(12, 35, len(daily))}, index=daily)
    fake_vix.index.name = "date"
    fake_move = pd.DataFrame({"value": rng.uniform(60, 120, len(daily))}, index=daily)
    fake_move.index.name = "date"
    fake_ovx = pd.DataFrame({"value": rng.uniform(20, 60, len(daily))}, index=daily)
    fake_ovx.index.name = "date"

    def _stub_vix(series_id, *_a, **_k):
        if series_id == "VIXCLS":
            return fake_vix
        return pd.DataFrame(columns=["value"])

    def _stub_move(*_a, **_k):
        return fake_move

    def _stub_ovx(*_a, **_k):
        return fake_ovx

    monkeypatch.setattr("apex.main_ensemble.fetch_fred_series", _stub_vix, raising=False)
    # main_ensemble imports inside the function, so monkeypatch the module-level
    monkeypatch.setattr("apex.data.fred_client.fetch_fred_series", _stub_vix)
    monkeypatch.setattr("apex.data.cross_asset_vol.fetch_move_index", _stub_move)
    monkeypatch.setattr("apex.data.cross_asset_vol.fetch_ovx", _stub_ovx)

    data_dict = {
        "SPY": {
            "exec_df": edf.copy(),
            "exec_df_holdout": edf.iloc[:50].copy(),
            "daily_df": None,
            "daily_df_holdout": None,
        }
    }
    cfg = {"cache_dir": "apex_cache", "options_gex": {"enabled": False}}
    out = main_ensemble.prepare_ensemble_data(data_dict, cfg)
    edf_out = out["SPY"]["exec_df"]
    for col in ("vix_pct", "move_pct", "ovx_pct"):
        assert col in edf_out.columns, f"missing column {col}"


def test_prepare_ensemble_data_handles_options_gex_disabled():
    from apex import main_ensemble
    cfg = {"cache_dir": "apex_cache", "options_gex": {"enabled": False}}
    data = _fixture_data_dict()
    # Without options_gex.enabled, GEX columns are not required
    out = main_ensemble.prepare_ensemble_data(data, cfg)
    assert "SPY" in out
