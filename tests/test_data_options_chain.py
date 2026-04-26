"""Tests for options-chain helpers used by Strategy 1 + Strategy 4."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def mock_chain():
    """Synthetic options chain — same shape as options_chain_sample.json fixture."""
    path = Path(__file__).parent / "fixtures" / "options_chain_sample.json"
    return json.loads(path.read_text())


def test_extract_call_put_walls(mock_chain, monkeypatch):
    from apex.data import options_chain
    from apex.data import options_gex
    monkeypatch.setattr(options_gex, "_fetch_chain",
                        lambda sym, dt: mock_chain)
    levels = options_chain.fetch_gex_levels("SPY", "2025-06-17", cache_dir=None)
    assert "call_wall" in levels
    assert "put_wall" in levels
    assert "gamma_flip" in levels
    assert isinstance(levels["call_wall"], float)
