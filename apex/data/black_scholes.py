"""Black-Scholes synthetic greeks for historical option chains.

Polygon Options Starter doesn't provide historical greeks -- only the live
snapshot does. For backtesting we compute synthetic greeks from historical
OHLC option prices + spot + DTE + risk-free rate.

Caveat: synthetic greeks won't perfectly match what dealers actually held
at the time (no dividend adjustment, single risk-free rate, no skew model
in IV). They are good enough for ranking / wall identification but not
for precise PnL attribution.
"""
import math
from typing import Optional

from scipy.stats import norm


# ~5y average 3-month T-bill yield. Could pull from FRED DGS3MO if needed.
RISK_FREE_RATE_DEFAULT = 0.045


def bs_price_and_vega(spot: float, strike: float, t: float, r: float,
                      sigma: float, is_call: bool):
    """Return (Black-Scholes price, vega) for one contract.

    Vega is dPrice/dSigma per unit (not per 1%). Returns (0,0) on degenerate inputs.
    """
    if t <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return 0.0, 0.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + sigma * sigma / 2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if is_call:
        price = spot * norm.cdf(d1) - strike * math.exp(-r * t) * norm.cdf(d2)
    else:
        price = strike * math.exp(-r * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    vega = spot * norm.pdf(d1) * sqrt_t
    return float(price), float(vega)


def implied_volatility(option_price: float, spot: float, strike: float,
                       time_to_expiry_years: float,
                       risk_free_rate: float = RISK_FREE_RATE_DEFAULT,
                       is_call: bool = True, max_iter: int = 100,
                       tol: float = 1e-5) -> Optional[float]:
    """Newton-Raphson IV solver. Returns None on non-convergence or bad inputs."""
    if (option_price <= 0 or spot <= 0 or strike <= 0
            or time_to_expiry_years <= 0):
        return None
    sigma = 0.30  # reasonable starting guess for equity index options
    for _ in range(max_iter):
        bs_price, vega = bs_price_and_vega(spot, strike, time_to_expiry_years,
                                           risk_free_rate, sigma, is_call)
        diff = bs_price - option_price
        if abs(diff) < tol:
            return float(sigma)
        if vega < 1e-10:
            return None
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 0.01
        if sigma > 5:
            return None
    return None


def bs_delta(spot: float, strike: float, t: float, r: float,
             sigma: float, is_call: bool) -> float:
    """Black-Scholes delta. Returns NaN on invalid inputs."""
    if t <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return float("nan")
    d1 = (math.log(spot / strike) + (r + sigma * sigma / 2) * t) / (sigma * math.sqrt(t))
    if is_call:
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)


def bs_gamma(spot: float, strike: float, t: float, r: float,
             sigma: float) -> float:
    """Black-Scholes gamma. Same formula for calls and puts.

    Returns NaN on invalid inputs.
    """
    if t <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return float("nan")
    d1 = (math.log(spot / strike) + (r + sigma * sigma / 2) * t) / (sigma * math.sqrt(t))
    return float(norm.pdf(d1) / (spot * sigma * math.sqrt(t)))
