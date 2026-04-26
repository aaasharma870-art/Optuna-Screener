"""Tests for EnsembleCombiner orchestrator."""
import numpy as np
import pandas as pd
import pytest


class _FakeStrategy:
    """Minimal strategy for combiner tests — fixed signals + size."""
    def __init__(self, name, signal_series, size_value=0.5):
        self.name = name
        self.data_requirements = []
        self._signal = signal_series
        self._size = size_value

    def compute_signals(self, data):
        n = len(self._signal)
        return pd.DataFrame({
            "entry_long":  self._signal.astype(bool),
            "entry_short": pd.Series([False] * n),
            "exit_long":   pd.Series([False] * n),
            "exit_short":  pd.Series([False] * n),
        })

    def compute_position_size(self, data, signals):
        return pd.Series([self._size] * len(self._signal))

    def get_tunable_params(self):
        return {}


def _make_data(n=100):
    return {
        "exec_df_1H": pd.DataFrame({
            "datetime": pd.date_range("2025-01-02", periods=n, freq="h"),
            "close": np.linspace(100, 105, n),
        }),
        "regime_state": pd.Series(["R2"] * n),
    }


def test_combiner_runs_with_two_strategies():
    from apex.ensemble.combiner import EnsembleCombiner
    n = 100
    sig_a = pd.Series([True if i % 10 == 0 else False for i in range(n)])
    sig_b = pd.Series([True if i % 7 == 0 else False for i in range(n)])
    strategies = [_FakeStrategy("vrp_gex_fade", sig_a, 0.5),
                  _FakeStrategy("opex_gravity", sig_b, 0.4)]
    combiner = EnsembleCombiner(strategies)
    result = combiner.run(_make_data(n))
    assert "portfolio_position" in result
    assert "weights" in result
    assert "trades" in result
    assert isinstance(result["portfolio_position"], pd.Series)
    assert len(result["portfolio_position"]) == n


def test_combiner_respects_30pct_cap():
    """No strategy should be allocated more than 30% of portfolio weight.

    Use 5 strategies so cap (0.30 * 5 = 1.50) is achievable while preserving
    sum=1. With non-trivial signals each strategy has measurable vol so risk-
    parity weights are well-defined; the cap must clamp any outliers.
    """
    from apex.ensemble.combiner import EnsembleCombiner
    rng = np.random.default_rng(7)
    n = 200
    # Use a non-trending price series so per-strategy returns are non-trivial
    data = _make_data(n)
    data["exec_df_1H"]["close"] = 100 + np.cumsum(rng.normal(0, 0.3, n))
    # Use UNKNOWN regime so the regime overlay does not tilt weights past the
    # risk-parity cap (overlay tilts apply *after* the cap).
    data["regime_state"] = pd.Series(["UNKNOWN"] * n)
    # Different signal cadences => different realized vols
    sigs = [pd.Series([i % k == 0 for i in range(n)]) for k in (3, 5, 7, 11, 13)]
    sizes = [0.50, 0.40, 0.30, 0.25, 0.20]
    names = ["vrp_gex_fade", "opex_gravity", "vix_term_structure",
             "vol_skew_arb", "smc_structural"]
    strategies = [_FakeStrategy(n_, s, sz)
                  for n_, s, sz in zip(names, sigs, sizes)]
    combiner = EnsembleCombiner(strategies, max_weight=0.30)
    result = combiner.run(data)
    for w in result["weights"].values():
        assert w <= 0.30 + 1e-9


def test_combiner_zeroes_weights_in_r4():
    from apex.ensemble.combiner import EnsembleCombiner
    n = 50
    strategies = [_FakeStrategy("vrp_gex_fade",
                                pd.Series([True] * n), 0.5)]
    data = _make_data(n)
    data["regime_state"] = pd.Series(["R4"] * n)
    combiner = EnsembleCombiner(strategies)
    result = combiner.run(data)
    # Final positions should be zero in R4
    assert (result["portfolio_position"].abs() < 1e-9).all()
