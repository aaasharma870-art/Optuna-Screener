"""Tests for OPEX calendar + pin-strike helpers."""
import pandas as pd

from apex.data.opex_calendar import find_pin_strike, is_opex_week


def test_third_friday_detection():
    """Third Friday week is OPEX week; other weeks are not."""
    # April 2026: 1st = Wed, 1st Friday = Apr 3, 3rd Friday = Apr 17.
    # The week of April 13-17 (Mon-Fri) should be OPEX week.
    assert is_opex_week(pd.Timestamp("2026-04-17")) is True  # Friday itself
    assert is_opex_week(pd.Timestamp("2026-04-13")) is True  # Monday
    assert is_opex_week(pd.Timestamp("2026-04-15")) is True  # Wednesday
    # Off-week
    assert is_opex_week(pd.Timestamp("2026-04-06")) is False
    assert is_opex_week(pd.Timestamp("2026-04-24")) is False
    # Different month: May 2026 third Friday = May 15.
    assert is_opex_week(pd.Timestamp("2026-05-15")) is True
    assert is_opex_week(pd.Timestamp("2026-05-08")) is False


def test_pin_strike_picks_highest_oi():
    """Pin strike should be the strike with highest total OI inside window."""
    chain = {
        "strikes": [
            {"strike": 395.0, "call_oi": 100, "put_oi": 200},
            {"strike": 400.0, "call_oi": 5000, "put_oi": 4000},  # highest combined
            {"strike": 405.0, "call_oi": 300, "put_oi": 100},
        ],
    }
    pin = find_pin_strike(chain, spot=400.0, window_pct=0.05)
    assert pin == 400.0


def test_window_excludes_far_strikes():
    """Strikes outside +/- window_pct should be excluded entirely."""
    chain = {
        "strikes": [
            # 50% OTM call wall — must be excluded by 5% window
            {"strike": 600.0, "call_oi": 999999, "put_oi": 999999},
            {"strike": 400.0, "call_oi": 1000, "put_oi": 1000},
            {"strike": 405.0, "call_oi": 500, "put_oi": 500},
        ],
    }
    pin = find_pin_strike(chain, spot=400.0, window_pct=0.05)
    assert pin == 400.0  # 600 excluded; 400 wins among in-window strikes
    # And: empty chain returns None
    assert find_pin_strike({"strikes": []}, spot=400.0) is None
    # No 'strikes' key at all returns None
    assert find_pin_strike({}, spot=400.0) is None
