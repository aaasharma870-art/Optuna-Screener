"""Borrow-fee model for short-side execution."""

TRADING_DAYS = 252


def borrow_fee(entry_price: float, annual_rate: float, days_held: float) -> float:
    """Compute borrow fee: entry_price * annual_rate * days_held / 252."""
    return entry_price * annual_rate * days_held / TRADING_DAYS


def borrow_fee_from_bars(entry_price: float, annual_rate: float,
                         bars_held: int, bars_per_day: int = 7) -> float:
    """Same math but takes bars_held; converts to days first."""
    days_held = bars_held / bars_per_day
    return borrow_fee(entry_price, annual_rate, days_held)


def lookup_borrow_rate(symbol: str, rates: dict) -> float:
    """Lookup borrow rate for symbol, falling back to 'default' key."""
    return rates.get(symbol, rates.get("default", 0.0))
