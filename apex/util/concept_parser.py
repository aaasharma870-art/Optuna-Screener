"""Concept parser and indicator registry."""


INDICATOR_REGISTRY = {
    "RSI": {
        "compute": "compute_rsi",
        "params": {"rsi_period": (5, 30), "rsi_oversold": (20, 40), "rsi_overbought": (60, 85)},
        "signal_type": "oscillator",
    },
    "MACD": {
        "compute": "compute_macd",
        "params": {"macd_fast": (8, 16), "macd_slow": (20, 35), "macd_signal": (5, 12)},
        "signal_type": "crossover",
    },
    "Bollinger": {
        "compute": "compute_bollinger",
        "params": {"boll_period": (14, 30), "boll_std": (1.5, 3.0)},
        "signal_type": "band",
    },
    "Stochastic": {
        "compute": "compute_stochastic",
        "params": {"stoch_k": (5, 21), "stoch_d": (3, 7)},
        "signal_type": "oscillator",
    },
    "OBV": {
        "compute": "compute_obv",
        "params": {"obv_ma_period": (10, 30)},
        "signal_type": "volume",
    },
    "ADX": {
        "compute": "compute_adx",
        "params": {"adx_period": (10, 25), "adx_threshold": (20, 35)},
        "signal_type": "trend_strength",
    },
    "CCI": {
        "compute": "compute_cci",
        "params": {"cci_period": (14, 30), "cci_oversold": (-150, -80), "cci_overbought": (80, 150)},
        "signal_type": "oscillator",
    },
    "WilliamsR": {
        "compute": "compute_williams_r",
        "params": {"willr_period": (7, 21), "willr_oversold": (-90, -70), "willr_overbought": (-30, -10)},
        "signal_type": "oscillator",
    },
    "Keltner": {
        "compute": "compute_keltner",
        "params": {"keltner_period": (14, 30), "keltner_mult": (1.0, 3.0)},
        "signal_type": "band",
    },
    "VolumeSurge": {
        "compute": "compute_volume_surge",
        "params": {"volume_surge_ma": (10, 30), "volume_surge_mult": (1.2, 3.0)},
        "signal_type": "volume",
    },
    "VWAP": {
        "compute": "compute_vwap",
        "params": {},
        "signal_type": "level",
    },
    "EMA_Cross": {
        "compute": "compute_ema",
        "params": {"ema_fast": (5, 15), "ema_slow": (18, 50)},
        "signal_type": "crossover",
    },
}


def parse_concept(concept_str):
    """
    Parse a human-readable strategy concept string into indicator bias weights.

    Examples:
      ``"mean reversion with volume confirmation"``
      ``"trend following momentum breakout"``

    Returns a dict mapping indicator names to float weights (0.0 to 2.0).
    """
    concept = concept_str.lower().strip()
    weights = {name: 1.0 for name in INDICATOR_REGISTRY}

    # Mean-reversion keywords boost oscillators, suppress trend
    mean_rev_kw = ["mean reversion", "revert", "bounce", "oversold", "dip", "pullback", "range"]
    if any(kw in concept for kw in mean_rev_kw):
        weights["RSI"] = 2.0
        weights["Bollinger"] = 2.0
        weights["Stochastic"] = 1.8
        weights["CCI"] = 1.5
        weights["WilliamsR"] = 1.5
        weights["ADX"] = 0.5
        weights["MACD"] = 0.6
        weights["EMA_Cross"] = 0.5

    # Trend-following keywords boost trend indicators
    trend_kw = ["trend", "momentum", "breakout", "follow", "directional"]
    if any(kw in concept for kw in trend_kw):
        weights["MACD"] = 2.0
        weights["ADX"] = 2.0
        weights["EMA_Cross"] = 2.0
        weights["Keltner"] = 1.5
        weights["RSI"] = 0.7
        weights["Stochastic"] = 0.5
        weights["Bollinger"] = 0.8

    # Volume keywords
    vol_kw = ["volume", "surge", "liquidity", "accumulation"]
    if any(kw in concept for kw in vol_kw):
        weights["VolumeSurge"] = 2.0
        weights["OBV"] = 2.0
        weights["VWAP"] = 1.5

    # Volatility keywords
    volat_kw = ["volatility", "squeeze", "expansion", "compress"]
    if any(kw in concept for kw in volat_kw):
        weights["Bollinger"] = 2.0
        weights["Keltner"] = 2.0
        weights["ADX"] = 1.5
        weights["CCI"] = 1.3

    # Scalp / intraday keywords
    scalp_kw = ["scalp", "intraday", "quick", "fast"]
    if any(kw in concept for kw in scalp_kw):
        weights["VWAP"] = 2.0
        weights["RSI"] = 1.5
        weights["Stochastic"] = 1.5
        weights["VolumeSurge"] = 1.8
        weights["EMA_Cross"] = 1.3

    # Swing keywords
    swing_kw = ["swing", "multi-day", "position", "hold"]
    if any(kw in concept for kw in swing_kw):
        weights["MACD"] = 1.8
        weights["ADX"] = 1.8
        weights["Bollinger"] = 1.3
        weights["EMA_Cross"] = 1.5
        weights["OBV"] = 1.5

    return weights
