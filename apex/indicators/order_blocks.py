"""Order block detector for SMC strategy."""
from typing import Dict, List

import pandas as pd


def detect_order_blocks(df: pd.DataFrame, min_body_ratio: float = 0.5) -> List[Dict]:
    """Detect bullish + bearish order blocks (3-bar patterns).

    Bullish OB: bar[i] = down close (red), bar[i+1] = inside or small body,
                bar[i+2] = strong up close > bar[i].open AND body_ratio > min_body_ratio.

    Returns list of records: {start_idx, end_idx, direction, low, high, mitigated_at_idx}.
    """
    out: List[Dict] = []
    n = len(df)
    if n < 3:
        return out

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    for i in range(n - 2):
        # Bullish OB
        if closes[i] < opens[i]:  # red
            range_2 = highs[i + 2] - lows[i + 2]
            if range_2 > 0:
                body_2 = closes[i + 2] - opens[i + 2]
                body_ratio = abs(body_2) / range_2
                if body_2 > 0 and closes[i + 2] > opens[i] and body_ratio > min_body_ratio:
                    out.append({
                        "start_idx": i,
                        "end_idx": i + 2,
                        "direction": "bullish",
                        "low": float(lows[i]),
                        "high": float(opens[i]),
                        "mitigated_at_idx": None,
                    })
        # Bearish OB (mirror)
        if closes[i] > opens[i]:  # green
            range_2 = highs[i + 2] - lows[i + 2]
            if range_2 > 0:
                body_2 = closes[i + 2] - opens[i + 2]
                body_ratio = abs(body_2) / range_2
                if body_2 < 0 and closes[i + 2] < opens[i] and body_ratio > min_body_ratio:
                    out.append({
                        "start_idx": i,
                        "end_idx": i + 2,
                        "direction": "bearish",
                        "low": float(opens[i]),
                        "high": float(highs[i]),
                        "mitigated_at_idx": None,
                    })

    # Mitigation pass: track when price returns into OB zone
    for ob in out:
        for j in range(ob["end_idx"] + 1, n):
            if ob["direction"] == "bullish" and lows[j] <= ob["high"]:
                ob["mitigated_at_idx"] = j
                break
            if ob["direction"] == "bearish" and highs[j] >= ob["low"]:
                ob["mitigated_at_idx"] = j
                break

    return out
