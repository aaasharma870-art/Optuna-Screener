"""Tests for Layer C walk-forward weight stability runner."""
import json

import numpy as np
import pandas as pd
import pytest


def test_layer_c_returns_dynamic_static_uplift():
    from apex.main_ensemble import run_layer_c_validation
    rng = np.random.default_rng(42)
    # Need >= warmup_months + 3 = 9 months. Use 30 months of 1H bars.
    # 30 calendar months of 1H bars - use a long enough date span
    dt = pd.date_range("2022-01-03", "2024-09-30", freq="h")
    n = len(dt)
    rets_a = pd.Series(rng.normal(0.0001, 0.002, n))
    rets_b = pd.Series(rng.normal(0.00005, 0.003, n))
    res = run_layer_c_validation({"vrp_gex_fade": rets_a, "opex_gravity": rets_b},
                                  pd.Series(dt), {})
    assert "static_sharpe" in res, f"got: {res}"
    assert "dynamic_sharpe" in res
    assert "uplift" in res
    assert res.get("layer_c_status") in ("PASS", "FAIL", "UNKNOWN")


def test_layer_c_handles_empty_inputs():
    from apex.main_ensemble import run_layer_c_validation
    res = run_layer_c_validation({}, pd.Series(dtype="datetime64[ns]"), {})
    assert "error" in res or res.get("n_months", 0) == 0


def test_layer_c_pass_threshold_uses_uplift_05():
    """When uplift >= 0.05 status is PASS; otherwise FAIL."""
    from apex.main_ensemble import run_layer_c_validation
    # Build per-strategy returns where the dynamic re-weighting helps
    rng = np.random.default_rng(7)
    # 30 calendar months of 1H bars - use a long enough date span
    dt = pd.date_range("2022-01-03", "2024-09-30", freq="h")
    n = len(dt)
    rets_steady = pd.Series(rng.normal(0.0002, 0.001, n))
    rets_noisy = pd.Series(rng.normal(-0.0001, 0.005, n))
    res = run_layer_c_validation(
        {"vrp_gex_fade": rets_steady, "opex_gravity": rets_noisy},
        pd.Series(dt), {})
    if "uplift" in res:
        if res["uplift"] >= 0.05:
            assert res["layer_c_status"] == "PASS"
        else:
            assert res["layer_c_status"] == "FAIL"
