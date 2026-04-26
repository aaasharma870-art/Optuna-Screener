"""Tests for Black-Scholes synthetic greeks helper."""
import math

import pytest

from apex.data.black_scholes import (
    bs_delta,
    bs_gamma,
    bs_price_and_vega,
    implied_volatility,
)


def test_atm_call_iv_recovery():
    """Price an ATM call at sigma=0.20, then back-solve IV — should recover."""
    spot, strike, t, r, sigma = 100.0, 100.0, 0.5, 0.045, 0.20
    price, _ = bs_price_and_vega(spot, strike, t, r, sigma, is_call=True)
    iv = implied_volatility(price, spot, strike, t, risk_free_rate=r,
                            is_call=True)
    assert iv is not None
    assert abs(iv - sigma) < 1e-3


def test_atm_put_iv_recovery():
    """Same recovery test but for a put."""
    spot, strike, t, r, sigma = 100.0, 100.0, 0.25, 0.045, 0.30
    price, _ = bs_price_and_vega(spot, strike, t, r, sigma, is_call=False)
    iv = implied_volatility(price, spot, strike, t, risk_free_rate=r,
                            is_call=False)
    assert iv is not None
    assert abs(iv - sigma) < 1e-3


def test_atm_call_delta_near_half():
    """ATM call delta should be close to 0.5 (slightly higher with positive r)."""
    delta = bs_delta(spot=100.0, strike=100.0, t=0.25, r=0.045,
                     sigma=0.20, is_call=True)
    assert 0.45 < delta < 0.65


def test_atm_put_delta_near_negative_half():
    """ATM put delta should be close to -0.5."""
    delta = bs_delta(spot=100.0, strike=100.0, t=0.25, r=0.045,
                     sigma=0.20, is_call=False)
    assert -0.55 < delta < -0.35


def test_far_otm_gamma_near_zero():
    """A deep OTM call far from spot should have ~0 gamma."""
    gamma = bs_gamma(spot=100.0, strike=200.0, t=0.25, r=0.045, sigma=0.20)
    assert gamma >= 0
    assert gamma < 1e-5


def test_atm_gamma_positive():
    """ATM gamma should be strictly positive."""
    gamma = bs_gamma(spot=100.0, strike=100.0, t=0.25, r=0.045, sigma=0.20)
    assert gamma > 0


def test_iv_returns_none_on_bad_inputs():
    assert implied_volatility(-1.0, 100.0, 100.0, 0.25) is None
    assert implied_volatility(5.0, 0.0, 100.0, 0.25) is None
    assert implied_volatility(5.0, 100.0, 100.0, 0.0) is None


def test_delta_gamma_nan_on_bad_inputs():
    assert math.isnan(bs_delta(0.0, 100.0, 0.25, 0.045, 0.2, True))
    assert math.isnan(bs_gamma(100.0, 100.0, -0.1, 0.045, 0.2))
