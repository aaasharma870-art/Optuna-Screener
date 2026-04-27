"""Tests for the StrategyBase universe screener."""
import json

import numpy as np
import pandas as pd

from apex.strategies.base import StrategyBase


class _BuyTrendStrategy(StrategyBase):
    name = "buy_trend_test"
    data_requirements = ["exec_df_1H"]

    def __init__(self, params=None):
        self.params = params or {}

    def compute_signals(self, data):
        n = len(data["exec_df_1H"])
        return pd.DataFrame({
            "entry_long": [False] + [True] + [False] * max(0, n - 2),
            "entry_short": [False] * n,
            "exit_long": [False] * n,
            "exit_short": [False] * n,
        })

    def compute_position_size(self, data, signals):
        pos = np.zeros(len(signals))
        pos[signals["entry_long"].cummax().values] = 1.0
        return pd.Series(pos)

    def get_tunable_params(self):
        return {}


def _df(start, periods=80, drift=0.1):
    close = 100 + np.arange(periods) * drift
    return pd.DataFrame({
        "datetime": pd.date_range(start, periods=periods, freq="h"),
        "open": close,
        "high": close + 0.25,
        "low": close - 0.25,
        "close": close,
        "volume": np.full(periods, 1_000_000),
    })


def test_load_sp500_symbols_normalizes_class_tickers(monkeypatch):
    from apex.screener import load_sp500_symbols

    monkeypatch.setattr(
        pd,
        "read_html",
        lambda url: [pd.DataFrame({"Symbol": ["BRK.B", "AAPL", "BF.B"]})],
    )

    assert load_sp500_symbols() == ["AAPL", "BF-B", "BRK-B"]


def test_screener_writes_ranked_csv_and_json(tmp_path):
    from apex.screener import run_strategy_universe_screener

    data = {
        "AAA": {"exec_df": _df("2025-01-01", 260, 0.08),
                "exec_df_holdout": _df("2025-02-01", 80, 0.05)},
        "BBB": {"exec_df": _df("2025-01-01", 260, -0.03),
                "exec_df_holdout": _df("2025-02-01", 80, -0.02)},
    }
    cfg = {
        "phase3_params": {"exec_timeframe": "1H"},
        "execution": {"bars_per_day_1h": 7},
        "screening": {"min_tune_trades": 1, "min_holdout_trades": 1},
    }

    res = run_strategy_universe_screener(
        _BuyTrendStrategy, data, cfg, tmp_path, n_trials=0, top_n=2,
    )

    assert res["n_symbols"] == 2
    assert res["top"][0]["symbol"] == "AAA"
    assert (tmp_path / "strategy_universe_screen.csv").exists()
    payload = json.loads((tmp_path / "strategy_universe_screen.json").read_text())
    assert payload["strategy"] == "buy_trend_test"
    assert len(payload["all"]) == 2
