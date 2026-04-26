"""Options-chain helpers for strategies 1 (gamma walls) and 4 (vol skew)."""
from pathlib import Path
from typing import Optional

from apex.data.options_gex import compute_gex_proxy, _fetch_chain as _fetch_chain_for_date


def fetch_gex_levels(symbol: str, as_of: str, cache_dir: Optional[Path]) -> dict:
    """Wrapper around compute_gex_proxy that returns a clean dict for strategy use.

    Returns the same shape as compute_gex_proxy: call_wall, put_wall, gamma_flip,
    vol_trigger, abs_gamma_strike.
    """
    return compute_gex_proxy(symbol, as_of, cache_dir)
