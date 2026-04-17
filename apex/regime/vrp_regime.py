"""R1/R2/R3/R4 regime classifier based on VRP and VIX term structure."""

import numpy as np
import pandas as pd


def compute_vrp_regime(df: pd.DataFrame, vix_series: pd.Series,
                       vxv_series: pd.Series,
                       vrp_pct_series: pd.Series) -> pd.Series:
    """Classify each bar into one of four regimes.

    ts_ratio = vix / vxv

    R1 (Suppressed Fade):
        ts_ratio < 0.95 AND vrp_pct > 70 AND vix < 25
        -> full-size fade trades

    R2 (Reduced Fade):
        ts_ratio < 0.95 AND vrp_pct in [30, 70]
        -> half-size fades OK

    R3 (Amplified Trend):
        ts_ratio >= 0.95 AND vrp_pct < 30
        -> trend trades, either direction

    R4 (No Trade / Crisis):
        Everything else -- backwardation + suppressed VRP, VIX > 30,
        neutral VRP in backwardation, or missing data.

    Missing data -> R4.

    Parameters
    ----------
    df : pd.DataFrame
        Bar data (used only for index alignment).
    vix_series : pd.Series
        VIX values aligned to df.index (forward-filled from daily).
    vxv_series : pd.Series
        VXV (3-month VIX) values aligned to df.index.
    vrp_pct_series : pd.Series
        VRP percentile (0-100) aligned to df.index.

    Returns
    -------
    pd.Series
        Regime labels {"R1", "R2", "R3", "R4"} indexed like df.
    """
    regime = pd.Series("R4", index=df.index)

    vix = vix_series.reindex(df.index)
    vxv = vxv_series.reindex(df.index)
    vrp_pct = vrp_pct_series.reindex(df.index)

    # Term structure ratio
    ts_ratio = vix / vxv

    # Masks for missing data
    has_data = vix.notna() & vxv.notna() & vrp_pct.notna() & (vxv != 0)

    contango = ts_ratio < 0.95
    backwardation = ts_ratio >= 0.95

    # R1: Suppressed Fade
    r1_mask = has_data & contango & (vrp_pct > 70) & (vix < 25)
    regime[r1_mask] = "R1"

    # R2: Reduced Fade
    r2_mask = has_data & contango & (vrp_pct >= 30) & (vrp_pct <= 70)
    regime[r2_mask] = "R2"

    # R3: Amplified Trend
    r3_mask = has_data & backwardation & (vrp_pct < 30)
    regime[r3_mask] = "R3"

    # Everything else stays R4 (default)

    return regime
