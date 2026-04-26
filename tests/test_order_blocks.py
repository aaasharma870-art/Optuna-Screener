"""Tests for order block detector."""
import numpy as np
import pandas as pd

from apex.indicators.order_blocks import detect_order_blocks


def test_bullish_ob_detected():
    """Red bar followed by an inside bar then a strong green bar = bullish OB."""
    df = pd.DataFrame({
        # bar 0: down close (red), open=105 close=100 (range 99..106)
        # bar 1: inside / small body
        # bar 2: strong up close > bar[0].open AND body_ratio > 0.5
        "open":  [105.0, 101.0, 102.0],
        "high":  [106.0, 103.0, 110.0],
        "low":   [ 99.0,  99.0, 101.5],
        "close": [100.0, 102.0, 109.0],
    })
    obs = detect_order_blocks(df, min_body_ratio=0.5)
    assert len(obs) == 1
    assert obs[0]["direction"] == "bullish"
    assert obs[0]["start_idx"] == 0
    assert obs[0]["end_idx"] == 2
    # Bullish OB zone: low = bar[0].low, high = bar[0].open
    assert obs[0]["low"] == 99.0
    assert obs[0]["high"] == 105.0


def test_bearish_ob_detected():
    """Green bar then inside then strong red = bearish OB."""
    df = pd.DataFrame({
        # bar 0: up close (green), open=100 close=105 (range 99..106)
        # bar 2: strong down close < bar[0].open AND body_ratio > 0.5
        "open":  [100.0, 104.0, 103.0],
        "high":  [106.0, 105.0, 104.0],
        "low":   [ 99.0,  102.0, 95.0],
        "close": [105.0, 103.0,  96.0],
    })
    obs = detect_order_blocks(df, min_body_ratio=0.5)
    assert len(obs) == 1
    assert obs[0]["direction"] == "bearish"


def test_mitigation_tracked():
    """When price returns into the OB zone after detection, mitigated_at_idx is set."""
    df = pd.DataFrame({
        "open":  [105.0, 101.0, 102.0, 108.0, 107.0, 104.0],
        "high":  [106.0, 103.0, 110.0, 109.0, 108.0, 106.0],
        "low":   [ 99.0,  99.0, 101.5, 107.0, 103.0, 100.0],  # bar5 wicks down to 100
        "close": [100.0, 102.0, 109.0, 108.0, 105.0, 102.0],
    })
    obs = detect_order_blocks(df, min_body_ratio=0.5)
    bullish = [ob for ob in obs if ob["direction"] == "bullish"]
    assert bullish, "Expected at least one bullish OB"
    # The first bullish OB has high=105.0; bar5 low=100 <= 105 -> mitigated
    first = bullish[0]
    assert first["mitigated_at_idx"] is not None
    assert first["mitigated_at_idx"] >= first["end_idx"] + 1


def test_no_false_positives_on_flat_data():
    """Constant OHLC bars yield no order blocks."""
    n = 30
    df = pd.DataFrame({
        "open":  [100.0] * n,
        "high":  [100.0] * n,
        "low":   [100.0] * n,
        "close": [100.0] * n,
    })
    obs = detect_order_blocks(df, min_body_ratio=0.5)
    assert obs == []
