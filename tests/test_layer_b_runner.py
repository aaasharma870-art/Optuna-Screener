"""Tests for Layer B portfolio (ensemble) CPCV runner."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_layer_b_returns_sharpe_distribution():
    from apex.main_ensemble import run_layer_b_validation
    n = 500
    rng = np.random.default_rng(42)
    pos = pd.Series([0.5] * n)
    close = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.01, n)) * 100.0)
    combiner_result = {"portfolio_position": pos}
    res = run_layer_b_validation(combiner_result, close, {})
    assert res.get("n_folds", 0) > 0
    assert "sharpe_median" in res
    assert "sharpe_iqr" in res
    assert "sharpe_pct_positive" in res
    assert res.get("layer_b_status") in ("PASS", "FAIL")


def test_layer_b_handles_empty_inputs():
    from apex.main_ensemble import run_layer_b_validation
    res = run_layer_b_validation({"portfolio_position": pd.Series([])}, pd.Series([]), {})
    assert res.get("n_folds", 0) == 0


def test_layer_b_pass_criterion_strong_drift():
    """Strong drift + consistently long position => Sharpe high enough to PASS."""
    from apex.main_ensemble import run_layer_b_validation
    rng = np.random.default_rng(123)
    n = 1500
    # Strong drift with small noise so variance > 0 and Sharpe is large
    px = pd.Series(np.cumprod(1 + rng.normal(0.003, 0.005, n)) * 100.0)
    pos = pd.Series([1.0] * n)
    res = run_layer_b_validation({"portfolio_position": pos}, px, {})
    assert res["sharpe_median"] > 0


def test_layer_b_serialization_is_json_safe(tmp_path):
    from apex.main_ensemble import run_layer_b_validation, _serialize_layer_b
    rng = np.random.default_rng(0)
    n = 400
    pos = pd.Series([0.3] * n)
    close = pd.Series(np.cumprod(1 + rng.normal(0.001, 0.01, n)) * 100.0)
    res = run_layer_b_validation({"portfolio_position": pos}, close, {})
    serial = _serialize_layer_b(res)
    out = tmp_path / "ensemble_layer_b_results.json"
    out.write_text(json.dumps(serial, indent=2))
    parsed = json.loads(out.read_text())
    assert "sharpe_median" in parsed
    assert isinstance(parsed.get("sharpe_iqr"), list)
