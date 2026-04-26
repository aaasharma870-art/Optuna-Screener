"""Tests for compute_skew_ratio."""
from apex.data.vol_skew import compute_skew_ratio


def test_symmetric_chain_ratio_near_one():
    """When IV(25d put) == IV(25d call), ratio should equal 1.0."""
    chain = {
        "expirations": [{
            "dte": 30,
            "contracts": [
                {"type": "call", "delta": 0.25, "iv": 0.20},
                {"type": "put",  "delta": -0.25, "iv": 0.20},
                # off-delta contracts shouldn't pollute the result
                {"type": "call", "delta": 0.10, "iv": 0.30},
                {"type": "put",  "delta": -0.10, "iv": 0.30},
            ],
        }],
    }
    ratio = compute_skew_ratio(chain, dte_target=30, delta_target=0.25)
    assert ratio is not None
    assert abs(ratio - 1.0) < 1e-9


def test_put_skewed_chain_ratio_above_one():
    """Put IV elevated vs call IV -> ratio > 1.0 (typical equity skew)."""
    chain = {
        "expirations": [{
            "dte": 30,
            "contracts": [
                {"type": "call", "delta": 0.25, "iv": 0.20},
                {"type": "put",  "delta": -0.25, "iv": 0.30},  # 50% richer puts
            ],
        }],
    }
    ratio = compute_skew_ratio(chain, dte_target=30, delta_target=0.25)
    assert ratio is not None
    assert ratio > 1.0
    assert abs(ratio - 1.5) < 1e-9


def test_missing_data_returns_none():
    """No chain, no expirations, missing put/call side -> None."""
    assert compute_skew_ratio(None) is None
    assert compute_skew_ratio({}) is None
    assert compute_skew_ratio({"expirations": []}) is None
    # Calls only
    chain_no_puts = {
        "expirations": [{"dte": 30, "contracts": [
            {"type": "call", "delta": 0.25, "iv": 0.20}]}],
    }
    assert compute_skew_ratio(chain_no_puts) is None
    # Missing IV
    chain_no_iv = {
        "expirations": [{"dte": 30, "contracts": [
            {"type": "call", "delta": 0.25, "iv": None},
            {"type": "put",  "delta": -0.25, "iv": 0.20}]}],
    }
    assert compute_skew_ratio(chain_no_iv) is None
