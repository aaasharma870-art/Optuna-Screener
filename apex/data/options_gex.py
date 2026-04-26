"""Options-derived gamma exposure (GEX) proxy.

Computes Call Wall, Put Wall, Gamma Flip, Vol Trigger and Abs-Gamma Strike
from a Polygon-style options chain snapshot. The chain fetch is isolated in
``_fetch_chain`` so tests can monkeypatch it without touching the network.

Spec: docs/superpowers/specs/2026-04-14-optuna-screener-overhaul-design.md sec 6.2
Phase 14 wires _fetch_chain to apex.data.polygon_options.build_chain_for_date
which uses real Polygon historical contract metadata + close prices and
synthesises greeks via Black-Scholes (Polygon Starter has no historical greeks).
"""
import json
from pathlib import Path
from typing import Optional

import pandas as pd

CONTRACT_SIZE = 100  # standard equity option multiplier


def _fetch_chain(symbol: str, as_of) -> dict:
    """Fetch a real Polygon options chain snapshot for backtesting.

    Uses ``apex.data.polygon_options.build_chain_for_date`` which fetches
    historical option contracts + close prices and computes synthetic
    greeks via Black-Scholes. Greeks are synthetic (see polygon_options
    docstring for caveats).

    Returns an empty-shape chain (no contracts) on errors so callers
    degrade gracefully to NaN dealer levels rather than crashing.
    """
    from apex.data.polygon_options import build_chain_for_date
    from apex.data.polygon_client import fetch_daily
    from apex.config import CACHE_DIR

    if isinstance(as_of, str):
        as_of_str = as_of
    else:
        as_of_str = pd.Timestamp(as_of).strftime("%Y-%m-%d")

    # Resolve spot from cached daily bars (most recent close on/before as_of).
    try:
        _, daily_df, _ = fetch_daily(symbol)
    except Exception:
        return {}
    if daily_df is None or daily_df.empty:
        return {}

    daily_df = daily_df.copy()
    daily_df["datetime"] = pd.to_datetime(daily_df["datetime"])
    target_date = pd.Timestamp(as_of_str).normalize()
    matching = daily_df[daily_df["datetime"].dt.normalize() == target_date]
    if matching.empty:
        matching = daily_df[daily_df["datetime"].dt.normalize() < target_date]
    if matching.empty:
        return {}
    spot = float(matching.iloc[-1]["close"])

    chain_dir = Path(CACHE_DIR) / "options_chain"
    return build_chain_for_date(symbol, as_of_str, spot, chain_dir)


def _aggregate_gex(chain: dict, spot: float) -> dict:
    """Pure helper: bucket each contract's GEX onto its strike.

    Returns ``{strike: (call_gex, put_gex)}`` where:
        contract_gex = OI * CONTRACT_SIZE * spot**2 * gamma * 0.01
    Calls are recorded with a positive sign, puts with a negative sign
    (dealers are short customer puts).
    """
    by_strike: dict = {}
    expiries = chain.get("expiries", [])
    for exp in expiries:
        for c in exp.get("contracts", []):
            strike = float(c["strike"])
            oi = float(c.get("open_interest", 0))
            gamma = float(c.get("gamma", 0.0))
            mag = oi * CONTRACT_SIZE * spot * spot * gamma * 0.01
            call_gex, put_gex = by_strike.get(strike, (0.0, 0.0))
            if c.get("type") == "call":
                call_gex += mag
            else:
                put_gex += -mag
            by_strike[strike] = (call_gex, put_gex)
    return by_strike


def compute_gex_proxy(symbol: str, as_of, cache_dir: Optional[Path]) -> dict:
    """Return the five canonical dealer-level metrics for ``symbol`` at ``as_of``.

    Output dict keys: ``call_wall``, ``put_wall``, ``gamma_flip``,
    ``vol_trigger``, ``abs_gamma_strike`` -- all ``float``.

    If the chain fetch returns nothing usable, every key maps to ``float('nan')``.
    """
    nan_result = {
        "call_wall": float("nan"),
        "put_wall": float("nan"),
        "gamma_flip": float("nan"),
        "vol_trigger": float("nan"),
        "abs_gamma_strike": float("nan"),
    }

    # Try cache first if a directory was provided
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "gex" / f"{symbol}_{as_of}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                if isinstance(cached, dict) and "call_wall" in cached:
                    return {k: float(cached[k]) for k in nan_result}
            except (json.JSONDecodeError, OSError, KeyError, ValueError):
                pass

    chain = _fetch_chain(symbol, as_of)
    if not chain:
        return nan_result

    spot = float(chain.get("spot") or 0.0)
    if spot <= 0:
        return nan_result

    by_strike = _aggregate_gex(chain, spot)
    if not by_strike:
        return nan_result

    strikes = sorted(by_strike.keys())
    call_gex = [by_strike[s][0] for s in strikes]
    put_gex = [by_strike[s][1] for s in strikes]
    net_gex = [c + p for c, p in zip(call_gex, put_gex)]

    # Call wall = strike with the largest positive call GEX
    call_wall_idx = max(range(len(strikes)), key=lambda i: call_gex[i])
    call_wall = float(strikes[call_wall_idx])

    # Put wall = strike with the largest |put GEX| (puts are negative)
    put_wall_idx = min(range(len(strikes)), key=lambda i: put_gex[i])
    put_wall = float(strikes[put_wall_idx])

    # Abs gamma strike = strike with the largest |net GEX|
    abs_idx = max(range(len(strikes)), key=lambda i: abs(net_gex[i]))
    abs_gamma_strike = float(strikes[abs_idx])

    # Gamma flip: linear interpolation of the zero-crossing of cumulative net GEX
    cum = 0.0
    cum_series = []
    for n in net_gex:
        cum += n
        cum_series.append(cum)

    gamma_flip = float("nan")
    for i in range(1, len(cum_series)):
        prev, cur = cum_series[i - 1], cum_series[i]
        if prev == 0:
            gamma_flip = float(strikes[i - 1])
            break
        if (prev < 0 < cur) or (prev > 0 > cur):
            denom = (cur - prev)
            if denom != 0:
                frac = -prev / denom
                gamma_flip = float(strikes[i - 1] + frac * (strikes[i] - strikes[i - 1]))
            else:
                gamma_flip = float(strikes[i])
            break

    # If no zero crossing, fall back to the strike whose cumulative is closest to zero
    if gamma_flip != gamma_flip:  # NaN check
        nearest_idx = min(range(len(cum_series)), key=lambda i: abs(cum_series[i]))
        gamma_flip = float(strikes[nearest_idx])

    vol_trigger = float(0.85 * gamma_flip)

    return {
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": gamma_flip,
        "vol_trigger": vol_trigger,
        "abs_gamma_strike": abs_gamma_strike,
    }
