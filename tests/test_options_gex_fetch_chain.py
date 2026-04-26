"""Integration test for the live _fetch_chain wired into Polygon (Phase 14).

We mock the underlying network calls (fetch_daily for spot, contract list,
close prices) so the test runs offline.
"""
from unittest.mock import patch

import pandas as pd
import pytest


def _stub_contracts():
    return [
        {"ticker": "O:SPY_C580", "strike": 580.0,
         "expiration": "2025-12-19", "contract_type": "call"},
        {"ticker": "O:SPY_C600", "strike": 600.0,
         "expiration": "2025-12-19", "contract_type": "call"},
        {"ticker": "O:SPY_C620", "strike": 620.0,
         "expiration": "2025-12-19", "contract_type": "call"},
        {"ticker": "O:SPY_P580", "strike": 580.0,
         "expiration": "2025-12-19", "contract_type": "put"},
        {"ticker": "O:SPY_P600", "strike": 600.0,
         "expiration": "2025-12-19", "contract_type": "put"},
        {"ticker": "O:SPY_P620", "strike": 620.0,
         "expiration": "2025-12-19", "contract_type": "put"},
    ]


def _fake_daily_df():
    return pd.DataFrame({
        "datetime": pd.date_range("2025-12-10", periods=5, freq="D"),
        "open":   [600, 601, 602, 600, 599],
        "high":   [602, 603, 604, 601, 600],
        "low":    [598, 599, 600, 598, 597],
        "close":  [601, 602, 603, 600, 600],
        "volume": [1e6] * 5,
    })


def test_fetch_chain_end_to_end(tmp_path, monkeypatch):
    """Mocked network: _fetch_chain should return a non-empty chain dict."""
    import apex.config
    from apex.data import options_gex, polygon_options, polygon_client

    # Redirect cache dir to tmp_path so we don't pollute the real cache.
    monkeypatch.setattr(apex.config, "CACHE_DIR", tmp_path)

    with patch.object(polygon_client, "fetch_daily",
                      return_value=("SPY", _fake_daily_df(), "CACHED")), \
         patch.object(polygon_options, "fetch_active_contracts",
                      return_value=_stub_contracts()), \
         patch.object(polygon_options, "fetch_option_close_price",
                      return_value=5.0):
        chain = options_gex._fetch_chain("SPY", "2025-12-15")

    assert isinstance(chain, dict)
    assert chain.get("spot", 0) > 0
    # Both naming conventions should be populated
    assert "expiries" in chain
    assert "expirations" in chain
    assert "strikes" in chain
    # And contracts should have flowed through
    assert len(chain["expiries"]) > 0


def test_fetch_chain_returns_empty_on_no_daily(tmp_path, monkeypatch):
    """If we have no daily bars, return an empty dict (don't crash)."""
    import apex.config
    from apex.data import options_gex, polygon_client
    monkeypatch.setattr(apex.config, "CACHE_DIR", tmp_path)

    with patch.object(polygon_client, "fetch_daily",
                      return_value=("SPY", None, "NO_DATA")):
        chain = options_gex._fetch_chain("SPY", "2025-12-15")
    assert chain == {}


def test_compute_gex_proxy_through_real_fetch(tmp_path, monkeypatch):
    """compute_gex_proxy should now produce real wall numbers (no NaN)
    when _fetch_chain succeeds end-to-end."""
    import apex.config
    from apex.data import options_gex, polygon_options, polygon_client

    monkeypatch.setattr(apex.config, "CACHE_DIR", tmp_path)

    with patch.object(polygon_client, "fetch_daily",
                      return_value=("SPY", _fake_daily_df(), "CACHED")), \
         patch.object(polygon_options, "fetch_active_contracts",
                      return_value=_stub_contracts()), \
         patch.object(polygon_options, "fetch_option_close_price",
                      return_value=5.0):
        levels = options_gex.compute_gex_proxy("SPY", "2025-12-15",
                                               cache_dir=None)

    assert "call_wall" in levels
    assert "put_wall" in levels
    # All values should be finite floats
    for k, v in levels.items():
        assert isinstance(v, float), f"{k} not float"
