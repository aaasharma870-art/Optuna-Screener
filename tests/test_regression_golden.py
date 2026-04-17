"""
Golden-snapshot regression test.

Runs the core backtest engine (indicators + regime + backtest + stats) on
fixture data with DEFAULT_ARCHITECTURE / DEFAULT_PARAMS. Compares output
against a saved golden snapshot. If the snapshot doesn't exist yet, the
first run generates it and skips.

This catches behavioral drift in:
  - Indicator computation
  - Regime detection
  - Entry scoring
  - Backtest engine (trade generation, P&L, stats)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"
GOLDEN_FILE = GOLDEN_DIR / "pipeline_legacy.json"


def _run_core_backtest():
    """Run full_backtest on SPY fixture with default architecture/params."""
    import apex

    fixtures = Path(__file__).resolve().parent / "fixtures"
    df_1h = pd.read_parquet(fixtures / "SPY_1H.parquet")
    df_daily = pd.read_parquet(fixtures / "SPY_daily.parquet")

    architecture = dict(apex.DEFAULT_ARCHITECTURE)
    params = dict(apex.DEFAULT_PARAMS)

    trades, stats = apex.full_backtest(df_1h, df_daily, architecture, params)

    # Build serializable snapshot
    snapshot = {
        "stats": stats,
        "trade_count": len(trades),
        "trade_pnls": [round(t["pnl_pct"], 6) for t in trades] if trades else [],
    }
    return snapshot


def _snapshots_match(actual, expected, tol=1e-6):
    """Compare two snapshot dicts with float tolerance."""
    errors = []

    # Compare trade count
    if actual["trade_count"] != expected["trade_count"]:
        errors.append(
            f"trade_count: {actual['trade_count']} != {expected['trade_count']}"
        )

    # Compare stats
    for key in expected["stats"]:
        a_val = actual["stats"].get(key)
        e_val = expected["stats"].get(key)
        if isinstance(e_val, (int, float)) and isinstance(a_val, (int, float)):
            if abs(a_val - e_val) > tol:
                errors.append(f"stats.{key}: {a_val} != {e_val} (delta={abs(a_val - e_val)})")
        elif isinstance(e_val, dict) and isinstance(a_val, dict):
            for sub_k in e_val:
                if a_val.get(sub_k) != e_val[sub_k]:
                    errors.append(f"stats.{key}.{sub_k}: {a_val.get(sub_k)} != {e_val[sub_k]}")
        else:
            if a_val != e_val:
                errors.append(f"stats.{key}: {a_val!r} != {e_val!r}")

    # Compare trade PnLs
    a_pnls = actual.get("trade_pnls", [])
    e_pnls = expected.get("trade_pnls", [])
    if len(a_pnls) != len(e_pnls):
        errors.append(f"trade_pnls length: {len(a_pnls)} != {len(e_pnls)}")
    else:
        for i, (a, e) in enumerate(zip(a_pnls, e_pnls)):
            if abs(a - e) > tol:
                errors.append(f"trade_pnls[{i}]: {a} != {e}")
                if len(errors) > 10:
                    errors.append("... (truncated)")
                    break

    return errors


def test_golden_snapshot():
    """Regression gate: core backtest output must match golden snapshot."""
    np.random.seed(42)
    actual = _run_core_backtest()

    if not GOLDEN_FILE.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        with open(GOLDEN_FILE, "w") as f:
            json.dump(actual, f, indent=2)
        pytest.skip(
            f"Golden snapshot created at {GOLDEN_FILE}. "
            "Re-run to verify equality."
        )

    with open(GOLDEN_FILE, "r") as f:
        expected = json.load(f)

    errors = _snapshots_match(actual, expected)
    if errors:
        msg = "Golden snapshot mismatch:\n" + "\n".join(f"  - {e}" for e in errors)
        pytest.fail(msg)
