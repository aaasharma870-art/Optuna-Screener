"""Tests for the MockPolygonClient fixture."""

import pytest


def test_mock_fetch_bars_returns_fixture(mock_polygon):
    """Mock fetch_bars returns 180-bar 1H fixture with expected columns."""
    import apex

    sym, df, status = apex.fetch_bars("SPY", "1H")
    assert len(df) == 180
    assert "close" in df.columns
    assert "volume" in df.columns
    assert status == "FIXTURE"


def test_mock_fetch_daily_returns_fixture(mock_polygon):
    """Mock fetch_daily returns 400-bar daily fixture with expected columns."""
    import apex

    sym, df, status = apex.fetch_daily("SPY")
    assert len(df) == 400
    assert "close" in df.columns
    assert status == "FIXTURE"


def test_mock_unknown_symbol_raises(mock_polygon):
    """Unknown symbol raises FileNotFoundError."""
    import apex

    with pytest.raises(FileNotFoundError):
        apex.fetch_bars("ZZZZZ", "1H")

    with pytest.raises(FileNotFoundError):
        apex.fetch_daily("ZZZZZ")
