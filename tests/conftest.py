"""Shared test fixtures for the Optuna Screener test suite."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ------------------------------------------------------------------
# Mock Polygon client
# ------------------------------------------------------------------

class MockPolygonClient:
    """Reads from parquet fixtures instead of hitting Polygon API."""

    def __init__(self, fixtures_dir: Path = FIXTURES_DIR):
        self.fixtures_dir = fixtures_dir

    def fetch_bars(self, symbol, timeframe="1H", start_date=None, end_date=None):
        fname = f"{symbol}_{timeframe}.parquet"
        path = self.fixtures_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"No fixture for {symbol} {timeframe}: {path}")
        df = pd.read_parquet(path)
        return symbol, df, "FIXTURE"

    def fetch_daily(self, symbol):
        path = self.fixtures_dir / f"{symbol}_daily.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No daily fixture for {symbol}: {path}")
        df = pd.read_parquet(path)
        return symbol, df, "FIXTURE"


_mock_client = MockPolygonClient()


@pytest.fixture
def mock_polygon(monkeypatch):
    """Monkeypatch apex.fetch_bars and apex.fetch_daily to use fixtures."""
    import apex

    monkeypatch.setattr(apex, "fetch_bars", _mock_client.fetch_bars)
    monkeypatch.setattr(apex, "fetch_daily", _mock_client.fetch_daily)
    return _mock_client


# ------------------------------------------------------------------
# Deterministic seed (autouse)
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Reset numpy random seed before every test for reproducibility."""
    np.random.seed(42)
    yield


# ------------------------------------------------------------------
# Tiny-budget config for fast Optuna tests
# ------------------------------------------------------------------

@pytest.fixture
def tiny_budget_cfg():
    """Minimal Optuna config for fast tests."""
    return {
        "optimization": {
            "arch_trials": 3,
            "inner_trials": 2,
            "deep_trials": 5,
            "walk_forward_oos_pct": 0.30,
            "fitness_is_weight": 0.4,
            "fitness_oos_weight": 0.6,
            "top_architectures_to_keep": 2,
            "max_symbols_to_optimize": 3,
            "final_holdout_pct": 0.25,
            "max_correlation": 0.70,
            "max_per_sector": 3,
            "robustness_threshold": 0.2,
        },
        "robustness": {
            "monte_carlo_sims": 50,
            "min_prob_profit": 0.5,
            "noise_injection_bar_jitter": 1,
            "noise_injection_price_pct": 5,
            "param_jitter_pct": 10,
            "min_robustness_score": 0.2,
        },
    }
