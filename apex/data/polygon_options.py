"""Polygon options chain ingest with synthetic-greek augmentation.

Polygon Options Starter exposes:
  - /v3/reference/options/contracts (historical contract metadata, as_of filter)
  - /v2/aggs/ticker/O:{contract}/range/1/day/{from}/{to} (historical OHLCV)
  - /v3/snapshot/options/{ticker} (live snapshot, includes greeks)

Historical greeks are NOT available on Starter. This module:
  1. Fetches the active contract list for `symbol` on `as_of`.
  2. Fetches each contract's closing price on `as_of`.
  3. Back-solves IV via Black-Scholes from price + spot + DTE.
  4. Computes delta/gamma analytically.
  5. Returns a chain dict shaped for both `_aggregate_gex` (expiries) AND
     compute_skew_ratio / find_pin_strike (expirations + strikes).

Caveats:
  - Synthetic greeks differ from dealer-held greeks (no dividend, no skew model).
  - Open interest is stubbed (Polygon Starter doesn't deliver historical OI cleanly).
    GEX magnitudes are therefore scaled by a uniform proxy; relative ranking of
    walls is still meaningful, absolute dollar gamma is not.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests

from apex.config import POLYGON_KEY, POLYGON_BASE, POLYGON_SLEEP
from apex.data.black_scholes import (
    RISK_FREE_RATE_DEFAULT,
    bs_delta,
    bs_gamma,
    implied_volatility,
)
from apex.logging_util import log


# Stub OI used until a real historical-OI source is wired in. Holding it
# uniform means the wall identification (argmax of gamma) ranks by gamma alone,
# which is still informative for pin / wall analysis.
_OI_STUB = 1000


def fetch_active_contracts(symbol: str, as_of: str,
                           cache_dir: Optional[Path] = None) -> List[dict]:
    """Fetch all option contracts for ``symbol`` that were active on ``as_of``.

    Returns list of dicts: ``{ticker, strike, expiration, contract_type}``.
    Uses Polygon's /v3/reference/options/contracts endpoint with as_of filter.
    Cached per (symbol, as_of) when ``cache_dir`` is provided.
    """
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"contracts_{symbol}_{as_of}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass

    contracts: List[dict] = []
    next_url = None
    endpoint = f"{POLYGON_BASE}/v3/reference/options/contracts"
    base_params = {
        "underlying_ticker": symbol,
        "as_of": as_of,
        "expired": "false",
        "limit": 1000,
        "apiKey": POLYGON_KEY,
    }

    page = 0
    while page < 20:
        url = next_url if next_url else endpoint
        if next_url:
            req_params = {"apiKey": POLYGON_KEY}
        else:
            req_params = base_params
        try:
            r = requests.get(url, params=req_params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException:
            break
        results = data.get("results", []) or []
        for c in results:
            contracts.append({
                "ticker": c.get("ticker"),
                "strike": c.get("strike_price"),
                "expiration": c.get("expiration_date"),
                "contract_type": c.get("contract_type"),  # 'call' | 'put'
            })
        next_url = data.get("next_url")
        if not next_url:
            break
        page += 1
        time.sleep(POLYGON_SLEEP)

    if cache_file is not None and contracts:
        try:
            cache_file.write_text(json.dumps(contracts))
        except Exception:
            pass
    return contracts


def fetch_option_close_price(option_ticker: str, as_of: str,
                             cache_dir: Optional[Path] = None) -> Optional[float]:
    """Return the closing price of ``option_ticker`` on ``as_of``, or None."""
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"opt_{option_ticker.replace(':', '_')}_{as_of}.json"
        if cache_file.exists():
            try:
                d = json.loads(cache_file.read_text())
                v = d.get("close")
                return float(v) if v is not None else None
            except Exception:
                pass

    endpoint = (f"{POLYGON_BASE}/v2/aggs/ticker/{option_ticker}"
                f"/range/1/day/{as_of}/{as_of}")
    params = {"adjusted": "true", "apiKey": POLYGON_KEY}
    try:
        r = requests.get(endpoint, params=params, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", []) or []
        if not results:
            if cache_file is not None:
                try:
                    cache_file.write_text(json.dumps({"close": None}))
                except Exception:
                    pass
            return None
        close = float(results[0].get("c", 0))
        if cache_file is not None:
            try:
                cache_file.write_text(json.dumps({"close": close}))
            except Exception:
                pass
        time.sleep(POLYGON_SLEEP)
        return close
    except requests.RequestException:
        return None


def fetch_option_open_interest(option_ticker: str, as_of: str,
                               cache_dir: Optional[Path] = None) -> int:
    """Return open interest for ``option_ticker`` on ``as_of``.

    NOTE: Polygon Options Starter doesn't expose historical OI directly.
    We stub a uniform value so GEX math doesn't divide by zero. Relative
    wall ranking is preserved (argmax over gamma), absolute dollar
    gamma is not. A real OI feed should replace this in production.
    """
    return _OI_STUB


def build_chain_for_date(symbol: str, as_of: str, spot: float,
                         cache_dir: Path,
                         dte_filter_max: int = 60,
                         strike_band_pct: float = 0.15,
                         risk_free_rate: float = RISK_FREE_RATE_DEFAULT) -> dict:
    """Build a complete options chain dict for ``symbol`` on ``as_of``.

    Returns a dict with both naming conventions used downstream:
      - ``expiries``    : consumed by apex.data.options_gex._aggregate_gex
      - ``expirations`` : consumed by apex.data.vol_skew.compute_skew_ratio
      - ``strikes``     : consumed by apex.data.opex_calendar.find_pin_strike
      - ``calls`` / ``puts`` : raw lists for any future per-side aggregation

    Filters: only contracts within +/- ``strike_band_pct`` of spot and
    DTE in (0, ``dte_filter_max``].

    Greeks are synthetic (BS) -- see module docstring for caveats.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"chain_{symbol}_{as_of}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    empty_chain = {
        "spot": float(spot),
        "as_of": as_of,
        "calls": [],
        "puts": [],
        "strikes": [],
        "expirations": [],
        "expiries": [],
    }

    contracts = fetch_active_contracts(symbol, as_of, cache_dir=cache_dir)
    if not contracts:
        log(f"No contracts found for {symbol} on {as_of}", "WARN")
        return empty_chain

    try:
        as_of_dt = datetime.strptime(as_of, "%Y-%m-%d").date()
    except ValueError:
        return empty_chain

    lo = (1 - strike_band_pct) * spot
    hi = (1 + strike_band_pct) * spot
    contracts_filtered = [c for c in contracts
                          if c.get("strike") is not None
                          and lo <= float(c["strike"]) <= hi]

    calls: List[dict] = []
    puts: List[dict] = []
    strikes_dict: dict = {}
    expirations_dict: dict = {}

    for c in contracts_filtered:
        try:
            exp_str = c.get("expiration")
            if not exp_str:
                continue
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - as_of_dt).days
            if dte <= 0 or dte > dte_filter_max:
                continue

            ticker = c.get("ticker")
            if not ticker:
                continue

            close_price = fetch_option_close_price(ticker, as_of,
                                                   cache_dir=cache_dir)
            if close_price is None or close_price <= 0:
                continue

            strike = float(c["strike"])
            t = dte / 365.0
            is_call = c.get("contract_type") == "call"

            iv = implied_volatility(close_price, spot, strike, t,
                                    risk_free_rate=risk_free_rate,
                                    is_call=is_call)
            if iv is None:
                continue

            delta = bs_delta(spot, strike, t, risk_free_rate, iv, is_call)
            gamma = bs_gamma(spot, strike, t, risk_free_rate, iv)
            oi = fetch_option_open_interest(ticker, as_of, cache_dir=cache_dir)

            ctype = "call" if is_call else "put"
            rec = {
                "type": ctype,
                "strike": strike,
                "strike_price": strike,
                "open_interest": oi,
                "iv": iv,
                "delta": delta,
                "gamma": gamma,
                "dte": dte,
                "greeks": {"delta": delta, "gamma": gamma, "iv": iv},
            }

            if is_call:
                calls.append(rec)
            else:
                puts.append(rec)

            slot = strikes_dict.setdefault(
                strike, {"strike": strike, "call_oi": 0, "put_oi": 0})
            if is_call:
                slot["call_oi"] += oi
            else:
                slot["put_oi"] += oi

            exp_slot = expirations_dict.setdefault(
                dte, {"dte": dte, "expiry": exp_str, "contracts": []})
            exp_slot["contracts"].append(rec)
        except Exception:
            continue

    # `expiries` mirrors `expirations` but uses the key `_aggregate_gex` reads.
    expirations_list = list(expirations_dict.values())
    expiries_list = [
        {"expiry": e.get("expiry"), "dte": e["dte"], "contracts": e["contracts"]}
        for e in expirations_list
    ]

    chain = {
        "spot": float(spot),
        "as_of": as_of,
        "calls": calls,
        "puts": puts,
        "strikes": list(strikes_dict.values()),
        "expirations": expirations_list,
        "expiries": expiries_list,
    }
    try:
        cache_file.write_text(json.dumps(chain))
    except Exception:
        pass
    return chain
