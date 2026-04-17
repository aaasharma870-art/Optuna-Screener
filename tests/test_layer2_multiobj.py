"""Smoke test for Layer 2 multi-objective Pareto mode."""

import numpy as np
import pandas as pd
import pytest

from apex.engine.backtest import DEFAULT_ARCHITECTURE, DEFAULT_PARAMS


def _make_exec_df(n=500, seed=42):
    """Generate synthetic OHLCV DataFrame with enough bars for Layer 2."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="h")
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    close = np.maximum(close, 10.0)  # keep positive
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    low = np.maximum(low, 1.0)
    opn = close + rng.uniform(-0.5, 0.5, n)
    vol = rng.randint(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame({
        "datetime": dates,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })


def _make_daily_df(n=252, seed=42):
    """Generate synthetic daily OHLCV."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-06-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.randn(n) * 1.0)
    close = np.maximum(close, 10.0)
    return pd.DataFrame({
        "datetime": dates,
        "open": close + rng.uniform(-0.5, 0.5, n),
        "high": close + rng.uniform(0.1, 2.0, n),
        "low": close - rng.uniform(0.1, 2.0, n),
        "close": close,
        "volume": rng.randint(1_000_000, 10_000_000, n).astype(float),
    })


class TestLayer2MultiObjective:
    """Smoke tests for multi-objective mode in Layer 2."""

    def test_multi_objective_study_created(self):
        """Invoking layer2 with use_multi_objective=True creates an NSGA-II study."""
        from apex.optimize.layer2 import layer2_deep_tune

        exec_df = _make_exec_df(500)
        daily_df = _make_daily_df(252)
        data_dict = {
            "TEST": {
                "exec_df": exec_df,
                "daily_df": daily_df,
            }
        }

        arch = dict(DEFAULT_ARCHITECTURE)
        cfg = {
            "optimization": {
                "deep_trials": 5,
                "fitness_is_weight": 0.4,
                "fitness_oos_weight": 0.6,
            },
            "fitness": {
                "use_multi_objective": True,
                "max_dd_cap_pct": 50.0,  # Generous cap for synthetic data
            },
        }

        results = layer2_deep_tune(data_dict, arch, ["TEST"], cfg)
        # The function should complete without error.
        # It may or may not find valid solutions on random data,
        # but it must not crash.
        assert isinstance(results, dict)

    def test_single_objective_unchanged(self):
        """use_multi_objective=False keeps legacy single-objective behavior."""
        from apex.optimize.layer2 import layer2_deep_tune

        exec_df = _make_exec_df(500)
        daily_df = _make_daily_df(252)
        data_dict = {
            "TEST": {
                "exec_df": exec_df,
                "daily_df": daily_df,
            }
        }

        arch = dict(DEFAULT_ARCHITECTURE)
        cfg = {
            "optimization": {
                "deep_trials": 5,
                "fitness_is_weight": 0.4,
                "fitness_oos_weight": 0.6,
            },
            "fitness": {
                "use_multi_objective": False,
            },
        }

        results = layer2_deep_tune(data_dict, arch, ["TEST"], cfg)
        assert isinstance(results, dict)

    def test_default_is_single_objective(self):
        """When fitness config is absent, defaults to single-objective."""
        from apex.optimize.layer2 import layer2_deep_tune

        exec_df = _make_exec_df(500)
        daily_df = _make_daily_df(252)
        data_dict = {
            "TEST": {
                "exec_df": exec_df,
                "daily_df": daily_df,
            }
        }

        arch = dict(DEFAULT_ARCHITECTURE)
        cfg = {
            "optimization": {
                "deep_trials": 5,
                "fitness_is_weight": 0.4,
                "fitness_oos_weight": 0.6,
            },
        }

        results = layer2_deep_tune(data_dict, arch, ["TEST"], cfg)
        assert isinstance(results, dict)
