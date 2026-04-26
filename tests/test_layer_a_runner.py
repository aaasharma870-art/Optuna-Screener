"""Tests for Layer A per-strategy CPCV runner."""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class _DummyStrategy:
    """Always-long strategy: always +0.5 position."""

    name = "dummy_long"
    data_requirements = ["exec_df_1H"]

    def compute_signals(self, data):
        n = len(data["exec_df_1H"])
        return pd.DataFrame({
            "entry_long":  pd.Series([True] * n),
            "entry_short": pd.Series([False] * n),
            "exit_long":   pd.Series([False] * n),
            "exit_short":  pd.Series([False] * n),
        })

    def compute_position_size(self, data, signals):
        return pd.Series([0.5] * len(data["exec_df_1H"]))

    def get_tunable_params(self):
        return {}


def _fixture_data():
    fix = Path(__file__).parent / "fixtures" / "SPY_1H.parquet"
    edf = pd.read_parquet(fix)
    return {"SPY": {"exec_df": edf, "daily_df": None,
                    "exec_df_holdout": None, "daily_df_holdout": None}}


def test_layer_a_returns_per_strategy_per_symbol_rows():
    from apex.main_ensemble import run_layer_a_validation
    rows = run_layer_a_validation([_DummyStrategy()], _fixture_data(), {})
    assert len(rows) == 1
    row = rows[0]
    assert row["strategy_name"] == "dummy_long"
    assert row["symbol"] == "SPY"
    assert "median_sharpe" in row
    assert "iqr_low" in row
    assert "iqr_high" in row
    assert "pct_positive" in row
    assert row["layer_a_status"] in ("PASS", "FAIL", "ERROR")


def test_layer_a_writes_csv_with_correct_columns(tmp_path):
    from apex.main_ensemble import run_layer_a_validation, write_layer_a_csv
    rows = run_layer_a_validation([_DummyStrategy()], _fixture_data(), {})
    out_csv = tmp_path / "strategy_layer_a_results.csv"
    write_layer_a_csv(rows, out_csv)
    assert out_csv.exists()
    with open(out_csv) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        assert "strategy_name" in cols
        assert "symbol" in cols
        assert "median_sharpe" in cols
        assert "layer_a_status" in cols
        records = list(reader)
        assert len(records) == 1


class _CrashStrategy:
    name = "crash_strat"
    data_requirements = []
    def compute_signals(self, data):
        raise RuntimeError("boom")
    def compute_position_size(self, data, signals):
        return pd.Series([])
    def get_tunable_params(self):
        return {}


def test_layer_a_records_error_for_crashing_strategy():
    from apex.main_ensemble import run_layer_a_validation
    rows = run_layer_a_validation([_CrashStrategy()], _fixture_data(), {})
    assert len(rows) == 1
    assert rows[0]["layer_a_status"] in ("ERROR", "FAIL")
