"""Cross-asset basket data fetching for momentum alignment."""

from apex.data.polygon_client import fetch_daily
from apex.logging_util import log

DEFAULT_BASKET = ["SPY", "QQQ", "GLD", "USO", "IEF"]


def fetch_basket(symbols=None):
    """
    Return dict[symbol] -> daily OHLCV DataFrame.

    Uses fetch_daily from polygon_client for each symbol.
    Symbols that fail to fetch are silently omitted.
    """
    if symbols is None:
        symbols = list(DEFAULT_BASKET)

    basket = {}
    for sym in symbols:
        _, df, status = fetch_daily(sym)
        if df is not None and len(df) > 0:
            basket[sym] = df
            log(f"  Basket {sym}: {status} ({len(df)} bars)")
        else:
            log(f"  Basket {sym}: no data ({status})", "WARN")
    return basket
