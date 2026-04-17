"""Sector classification map for known symbols."""

SECTOR_MAP = {
    # Broad market ETFs
    "SPY": "Index", "SPX": "Index", "QQQ": "Index", "IWM": "Index", "DIA": "Index",
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "META": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "NFLX": "Technology", "CRM": "Technology", "ADBE": "Technology",
    "INTC": "Technology", "AVGO": "Technology", "MU": "Technology",
    "QCOM": "Technology", "ORCL": "Technology",
    # Semiconductors
    "SMH": "Semiconductors", "SOXX": "Semiconductors",
    "TSM": "Semiconductors", "ASML": "Semiconductors",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "C": "Financials", "WFC": "Financials",
    "XLF": "Financials", "V": "Financials", "MA": "Financials",
    # Energy
    "XLE": "Energy", "XOM": "Energy", "CVX": "Energy",
    "COP": "Energy", "SLB": "Energy",
    # Healthcare
    "XLV": "Healthcare", "JNJ": "Healthcare", "UNH": "Healthcare",
    "PFE": "Healthcare", "LLY": "Healthcare",
    # Consumer
    "AMZN": "Consumer", "TSLA": "Consumer", "WMT": "Consumer",
    "HD": "Consumer", "COST": "Consumer", "MCD": "Consumer",
    "DIS": "Consumer",
    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "HON": "Industrials",
    # Materials / Metals
    "GDX": "Materials", "GLD": "Materials", "SLV": "Materials",
    # Communication
    "GOOG": "Communication", "T": "Communication", "VZ": "Communication",
}
