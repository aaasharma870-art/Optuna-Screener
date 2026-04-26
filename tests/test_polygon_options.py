"""Tests for apex.data.polygon_options (chain builder + fetchers)."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest


# -- fetch_active_contracts ---------------------------------------------------

class _FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


def test_fetch_active_contracts_returns_list(tmp_path):
    from apex.data import polygon_options
    payload = {
        "results": [
            {"ticker": "O:SPY251219C00600000", "strike_price": 600.0,
             "expiration_date": "2025-12-19", "contract_type": "call"},
            {"ticker": "O:SPY251219P00600000", "strike_price": 600.0,
             "expiration_date": "2025-12-19", "contract_type": "put"},
        ]
    }
    with patch.object(polygon_options.requests, "get",
                      return_value=_FakeResp(payload)):
        out = polygon_options.fetch_active_contracts("SPY", "2025-12-15",
                                                     cache_dir=tmp_path)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["strike"] == 600.0
    assert out[0]["contract_type"] == "call"


def test_fetch_active_contracts_uses_cache(tmp_path):
    from apex.data import polygon_options
    cached = [{"ticker": "X", "strike": 1.0, "expiration": "2025-12-19",
               "contract_type": "call"}]
    cache_file = Path(tmp_path) / "contracts_SPY_2025-12-15.json"
    cache_file.write_text(json.dumps(cached))
    # Should NOT call requests if cache hits
    with patch.object(polygon_options.requests, "get",
                      side_effect=AssertionError("should not be called")):
        out = polygon_options.fetch_active_contracts("SPY", "2025-12-15",
                                                     cache_dir=tmp_path)
    assert out == cached


def test_fetch_active_contracts_handles_request_error(tmp_path):
    from apex.data import polygon_options
    import requests
    with patch.object(polygon_options.requests, "get",
                      side_effect=requests.RequestException("boom")):
        out = polygon_options.fetch_active_contracts("SPY", "2025-12-15",
                                                     cache_dir=tmp_path)
    assert out == []


# -- fetch_option_close_price -------------------------------------------------

def test_fetch_option_close_price_returns_float(tmp_path):
    from apex.data import polygon_options
    payload = {"results": [{"c": 4.25}]}
    with patch.object(polygon_options.requests, "get",
                      return_value=_FakeResp(payload)):
        v = polygon_options.fetch_option_close_price(
            "O:SPY251219C00600000", "2025-12-15", cache_dir=tmp_path)
    assert isinstance(v, float)
    assert v == 4.25


def test_fetch_option_close_price_returns_none_on_empty(tmp_path):
    from apex.data import polygon_options
    payload = {"results": []}
    with patch.object(polygon_options.requests, "get",
                      return_value=_FakeResp(payload)):
        v = polygon_options.fetch_option_close_price(
            "O:SPY251219C00600000", "2025-12-15", cache_dir=tmp_path)
    assert v is None


def test_fetch_option_close_price_cache_hit(tmp_path):
    from apex.data import polygon_options
    cache_file = Path(tmp_path) / "opt_O_SPY251219C00600000_2025-12-15.json"
    cache_file.write_text(json.dumps({"close": 7.5}))
    with patch.object(polygon_options.requests, "get",
                      side_effect=AssertionError("should not call")):
        v = polygon_options.fetch_option_close_price(
            "O:SPY251219C00600000", "2025-12-15", cache_dir=tmp_path)
    assert v == 7.5


# -- build_chain_for_date -----------------------------------------------------

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
        # Out-of-band strike (should be filtered out by 15% band at spot=600)
        {"ticker": "O:SPY_C800", "strike": 800.0,
         "expiration": "2025-12-19", "contract_type": "call"},
    ]


def test_build_chain_for_date_returns_required_keys(tmp_path):
    from apex.data import polygon_options

    def fake_close(ticker, as_of, cache_dir=None):
        # Plausible prices that will be invertible
        return 5.0

    with patch.object(polygon_options, "fetch_active_contracts",
                      return_value=_stub_contracts()), \
         patch.object(polygon_options, "fetch_option_close_price",
                      side_effect=fake_close):
        chain = polygon_options.build_chain_for_date(
            "SPY", "2025-12-15", spot=600.0, cache_dir=tmp_path)

    for k in ("spot", "calls", "puts", "strikes", "expirations", "expiries"):
        assert k in chain
    assert chain["spot"] == 600.0
    # 800-strike should be filtered out by ATM ±15% band
    all_strikes = sorted({s["strike"] for s in chain["strikes"]})
    assert 800.0 not in all_strikes
    # We should have at least the 580/600/620 strikes
    assert set(all_strikes).issuperset({580.0, 600.0, 620.0})


def test_build_chain_for_date_skips_when_no_contracts(tmp_path):
    from apex.data import polygon_options
    with patch.object(polygon_options, "fetch_active_contracts",
                      return_value=[]):
        chain = polygon_options.build_chain_for_date(
            "SPY", "2025-12-15", spot=600.0, cache_dir=tmp_path)
    assert chain["calls"] == []
    assert chain["puts"] == []
    assert chain["expiries"] == []


def test_build_chain_for_date_handles_bad_price(tmp_path):
    """Contracts with no close price (None) should be skipped, not crash."""
    from apex.data import polygon_options
    with patch.object(polygon_options, "fetch_active_contracts",
                      return_value=_stub_contracts()), \
         patch.object(polygon_options, "fetch_option_close_price",
                      return_value=None):
        chain = polygon_options.build_chain_for_date(
            "SPY", "2025-12-15", spot=600.0, cache_dir=tmp_path)
    assert chain["calls"] == []
    assert chain["puts"] == []


def test_build_chain_for_date_uses_cache(tmp_path):
    from apex.data import polygon_options
    cached = {"spot": 600.0, "as_of": "2025-12-15", "calls": [],
              "puts": [], "strikes": [], "expirations": [], "expiries": []}
    cache_file = Path(tmp_path) / "chain_SPY_2025-12-15.json"
    cache_file.write_text(json.dumps(cached))
    with patch.object(polygon_options, "fetch_active_contracts",
                      side_effect=AssertionError("should not call")):
        chain = polygon_options.build_chain_for_date(
            "SPY", "2025-12-15", spot=600.0, cache_dir=tmp_path)
    assert chain["as_of"] == "2025-12-15"


def test_build_chain_for_date_filters_expired_contracts(tmp_path):
    """DTE <= 0 and DTE > dte_filter_max contracts should be excluded."""
    from apex.data import polygon_options
    contracts = [
        # Already expired
        {"ticker": "X1", "strike": 600.0, "expiration": "2025-12-10",
         "contract_type": "call"},
        # Way out (>60 DTE default)
        {"ticker": "X2", "strike": 600.0, "expiration": "2026-06-19",
         "contract_type": "call"},
        # In-window
        {"ticker": "X3", "strike": 600.0, "expiration": "2026-01-15",
         "contract_type": "call"},
    ]
    with patch.object(polygon_options, "fetch_active_contracts",
                      return_value=contracts), \
         patch.object(polygon_options, "fetch_option_close_price",
                      return_value=5.0):
        chain = polygon_options.build_chain_for_date(
            "SPY", "2025-12-15", spot=600.0, cache_dir=tmp_path)
    # Only X3 (DTE ~31) should make it
    assert len(chain["calls"]) == 1


def test_fetch_option_open_interest_returns_int():
    from apex.data import polygon_options
    v = polygon_options.fetch_option_open_interest("O:X", "2025-12-15")
    assert isinstance(v, int)
    assert v > 0
