"""Tests for the ensemble HTML report generator."""
import json
from pathlib import Path

import pandas as pd
import pytest


_TAB_LABELS = [
    "Headline",
    "Per-Strategy Contributions",
    "Equity Curves",
    "Regime Breakdown",
    "CPCV Distribution",
    "Walk-Forward Weights",
    "Layer A Results",
]


def _make_results():
    return {
        "primary_symbol": "SPY",
        "weights": {"vrp_gex_fade": 0.30, "opex_gravity": 0.20, "vix_term_structure": 0.20,
                    "vol_skew_arb": 0.15, "smc_structural": 0.15},
        "strategies": ["vrp_gex_fade", "opex_gravity", "vix_term_structure",
                       "vol_skew_arb", "smc_structural", "cross_asset_vol_overlay"],
        "layer_a_rows": [
            {"strategy_name": "vrp_gex_fade", "symbol": "SPY",
             "n_folds": 28, "median_sharpe": 0.42,
             "iqr_low": -0.15, "iqr_high": 0.95,
             "pct_positive": 0.61, "layer_a_status": "PASS"},
            {"strategy_name": "opex_gravity", "symbol": "SPY",
             "n_folds": 28, "median_sharpe": -0.12,
             "iqr_low": -0.45, "iqr_high": 0.30,
             "pct_positive": 0.43, "layer_a_status": "FAIL"},
        ],
        "layer_a_by_strategy": {"vrp_gex_fade": "PASS", "opex_gravity": "FAIL"},
        "layer_b": {
            "n_folds": 28,
            "oos_sharpes": [0.5, 0.8, 1.2, 0.3, -0.2, 1.1, 0.7, 0.9],
            "sharpe_median": 0.75,
            "sharpe_iqr": [0.3, 1.1],
            "sharpe_pct_positive": 0.875,
            "layer_b_status": "PASS",
            "portfolio_returns": [0.001, -0.002, 0.003, 0.0005, -0.001, 0.002] * 30,
        },
        "layer_c": {
            "static_sharpe": 0.85, "dynamic_sharpe": 1.05,
            "uplift": 0.20, "n_months": 18, "layer_c_status": "PASS",
        },
        "ref_close": [100.0 + i * 0.01 for i in range(180)],
        "ref_dt": [str(d) for d in pd.date_range("2024-01-02", periods=180, freq="h")],
        "portfolio_position": [0.5] * 180,
        "per_strategy_positions": {
            "vrp_gex_fade": [0.5] * 180,
            "opex_gravity": [-0.3] * 180,
        },
        "current_regime": "R2",
        "run_info": {"timestamp": "20260426_120000", "concept": "ensemble"},
    }


def test_generate_ensemble_report_creates_file(tmp_path):
    from apex.report.ensemble_report import generate_ensemble_report
    out = generate_ensemble_report(_make_results(), str(tmp_path))
    p = Path(out)
    assert p.exists()
    assert p.suffix == ".html"
    assert p.stat().st_size > 1000


def test_generate_ensemble_report_contains_all_seven_tab_labels(tmp_path):
    from apex.report.ensemble_report import generate_ensemble_report
    out = generate_ensemble_report(_make_results(), str(tmp_path))
    text = Path(out).read_text(encoding="utf-8")
    for label in _TAB_LABELS:
        assert label in text, f"tab label '{label}' missing"


def test_generate_ensemble_report_embeds_plotly_traces(tmp_path):
    from apex.report.ensemble_report import generate_ensemble_report
    out = generate_ensemble_report(_make_results(), str(tmp_path))
    text = Path(out).read_text(encoding="utf-8")
    # Plotly CDN script
    assert "plotly" in text.lower()
    # Expect multiple Plotly.newPlot calls
    assert text.count("Plotly.newPlot") >= 4
    # Expect at least one JSON-encoded sharpe value to validate
    assert "ensemble_cpcv_div" in text
    assert "ensemble_equity_div" in text


def test_generate_ensemble_report_handles_empty_layers(tmp_path):
    from apex.report.ensemble_report import generate_ensemble_report
    minimal = {
        "primary_symbol": "SPY",
        "weights": {},
        "strategies": [],
        "layer_a_rows": [],
        "layer_a_by_strategy": {},
        "layer_b": {},
        "layer_c": {},
        "ref_close": [],
        "ref_dt": [],
        "portfolio_position": [],
        "per_strategy_positions": {},
        "current_regime": "UNKNOWN",
        "run_info": {},
    }
    out = generate_ensemble_report(minimal, str(tmp_path))
    assert Path(out).exists()
