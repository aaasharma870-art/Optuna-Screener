"""Vol skew computation from options chain."""
from typing import Optional


def compute_skew_ratio(chain: dict, dte_target: int = 30,
                       delta_target: float = 0.25) -> Optional[float]:
    """Return IV(25-delta put) / IV(25-delta call) for chain at given DTE.

    Inputs:
      chain: dict with 'expirations' (each has 'dte' and 'contracts' list).
             Each contract has 'type' ('call' or 'put'), 'delta', 'iv'.
    Returns:
      ratio float, or None if either side missing.
    """
    if not chain or "expirations" not in chain:
        return None

    # Find expiry closest to dte_target
    exps = chain["expirations"]
    if not exps:
        return None
    best_exp = min(exps, key=lambda e: abs(e.get("dte", 9999) - dte_target))
    contracts = best_exp.get("contracts", [])

    # Find 25-delta put and call (closest to target delta)
    puts = [c for c in contracts if c.get("type") == "put"]
    calls = [c for c in contracts if c.get("type") == "call"]

    if not puts or not calls:
        return None

    put_25d = min(puts, key=lambda c: abs(abs(c.get("delta", 0)) - delta_target))
    call_25d = min(calls, key=lambda c: abs(abs(c.get("delta", 0)) - delta_target))

    iv_put = put_25d.get("iv")
    iv_call = call_25d.get("iv")
    if iv_put is None or iv_call is None or iv_call <= 1e-9:
        return None

    return float(iv_put) / float(iv_call)
