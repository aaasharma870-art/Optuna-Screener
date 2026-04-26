"""Options-derived gamma exposure (GEX) proxy.

Computes Call Wall, Put Wall, Gamma Flip, Vol Trigger and Abs-Gamma Strike
from a Polygon-style options chain snapshot. The chain fetch is isolated in
``_fetch_chain`` so tests can monkeypatch it without touching the network.

Spec: docs/superpowers/specs/2026-04-14-optuna-screener-overhaul-design.md sec 6.2
"""
import json
from pathlib import Path
from typing import Optional

CONTRACT_SIZE = 100  # standard equity option multiplier


def _fetch_chain(symbol: str, as_of) -> dict:
    """Fetch the options chain for ``symbol`` as of ``as_of``.

    The default implementation is a stub that raises so production callers
    must either:
      * monkeypatch this function in tests, or
      * pre-populate a JSON cache on disk and read via ``compute_gex_proxy``.

    Future Phase 1-completion work will swap this for a real Polygon fetch.
    Until then strategies receive gamma walls as pre-merged columns on
    ``exec_df_1H`` (see ``apex.data.dealer_levels.ingest_flux_points``).
    """
    raise NotImplementedError(
        "options_gex._fetch_chain is a stub. Monkeypatch in tests or pre-cache."
    )


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
