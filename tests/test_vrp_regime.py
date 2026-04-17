"""Tests for apex.regime.vrp_regime."""

import numpy as np
import pandas as pd
import pytest

from apex.regime.vrp_regime import compute_vrp_regime


def _make_df(n=1):
    """Create a minimal DataFrame with n rows."""
    return pd.DataFrame({"close": [100.0] * n}, index=range(n))


class TestVRPRegime:
    def test_r1_threshold(self):
        """R1: ts_ratio=0.94, vrp_pct=71, vix=24 -> R1."""
        df = _make_df(1)
        vix = pd.Series([24.0])
        vxv = pd.Series([24.0 / 0.94])  # ts_ratio = 0.94
        vrp_pct = pd.Series([71.0])
        regime = compute_vrp_regime(df, vix, vxv, vrp_pct)
        assert regime.iloc[0] == "R1"

    def test_r2_threshold(self):
        """R2: ts_ratio=0.94, vrp_pct=50 -> R2."""
        df = _make_df(1)
        vix = pd.Series([20.0])
        vxv = pd.Series([20.0 / 0.94])  # ts_ratio = 0.94
        vrp_pct = pd.Series([50.0])
        regime = compute_vrp_regime(df, vix, vxv, vrp_pct)
        assert regime.iloc[0] == "R2"

    def test_r3_threshold(self):
        """R3: ts_ratio=0.96, vrp_pct=29 -> R3."""
        df = _make_df(1)
        vix = pd.Series([20.0])
        vxv = pd.Series([20.0 / 0.96])  # ts_ratio = 0.96
        vrp_pct = pd.Series([29.0])
        regime = compute_vrp_regime(df, vix, vxv, vrp_pct)
        assert regime.iloc[0] == "R3"

    def test_r4_missing_data(self):
        """R4 for missing data."""
        df = _make_df(1)
        vix = pd.Series([np.nan])
        vxv = pd.Series([20.0])
        vrp_pct = pd.Series([50.0])
        regime = compute_vrp_regime(df, vix, vxv, vrp_pct)
        assert regime.iloc[0] == "R4"

    def test_r4_high_vix_contango(self):
        """R4 for VIX > 30 with contango (ts_ratio < 0.95, vrp_pct > 70 but vix > 25)."""
        df = _make_df(1)
        vix = pd.Series([32.0])
        vxv = pd.Series([32.0 / 0.90])  # contango
        vrp_pct = pd.Series([75.0])
        # contango + vrp_pct > 70 but vix >= 25 -> NOT R1, falls to R4
        regime = compute_vrp_regime(df, vix, vxv, vrp_pct)
        assert regime.iloc[0] == "R4"

    def test_r4_backwardation_neutral_vrp(self):
        """R4 for backwardation with neutral VRP (not < 30)."""
        df = _make_df(1)
        vix = pd.Series([22.0])
        vxv = pd.Series([22.0 / 0.98])  # ts_ratio = 0.98 >= 0.95
        vrp_pct = pd.Series([50.0])  # not < 30
        regime = compute_vrp_regime(df, vix, vxv, vrp_pct)
        assert regime.iloc[0] == "R4"

    def test_multiple_bars(self):
        """Test with multiple bars yielding different regimes."""
        df = _make_df(3)
        # Bar 0: R1 (contango, high vrp, low vix)
        # Bar 1: R3 (backwardation, low vrp)
        # Bar 2: R4 (missing)
        vix = pd.Series([20.0, 20.0, np.nan])
        vxv = pd.Series([20.0 / 0.90, 20.0 / 0.96, 25.0])
        vrp_pct = pd.Series([80.0, 25.0, 50.0])
        regime = compute_vrp_regime(df, vix, vxv, vrp_pct)
        assert regime.iloc[0] == "R1"
        assert regime.iloc[1] == "R3"
        assert regime.iloc[2] == "R4"

    def test_r2_boundary_vrp_30(self):
        """R2 at exact vrp_pct=30 boundary (inclusive)."""
        df = _make_df(1)
        vix = pd.Series([20.0])
        vxv = pd.Series([20.0 / 0.90])
        vrp_pct = pd.Series([30.0])
        regime = compute_vrp_regime(df, vix, vxv, vrp_pct)
        assert regime.iloc[0] == "R2"

    def test_r2_boundary_vrp_70(self):
        """R2 at exact vrp_pct=70 boundary (inclusive)."""
        df = _make_df(1)
        vix = pd.Series([20.0])
        vxv = pd.Series([20.0 / 0.90])
        vrp_pct = pd.Series([70.0])
        regime = compute_vrp_regime(df, vix, vxv, vrp_pct)
        assert regime.iloc[0] == "R2"
