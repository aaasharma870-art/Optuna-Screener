"""CSV and JSON report generation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from apex.logging_util import log
from apex.util.sector_map import SECTOR_MAP


def generate_trades_csv(all_trades, output_dir):
    """Export all trades to CSV."""
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    path = od / "trades.csv"
    if not all_trades:
        log("No trades to export")
        return str(path)
    df = pd.DataFrame(all_trades)
    df.to_csv(path, index=False)
    log(f"Trades CSV saved: {path}")
    return str(path)


def generate_summary_csv(results, output_dir):
    """Export per-symbol summary to CSV."""
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    path = od / "summary.csv"
    per_symbol = results.get("per_symbol", {})
    rows = []
    for sym, data in per_symbol.items():
        s = data.get("stats", {})
        row = {"symbol": sym, "sector": SECTOR_MAP.get(sym, "Unknown")}
        row.update(s)
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
    log(f"Summary CSV saved: {path}")
    return str(path)


def generate_parameters_json(results, architecture, output_dir):
    """Export optimized parameters and architecture to JSON."""
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    path = od / "parameters.json"

    per_symbol = results.get("per_symbol", {})
    params_out = {}
    for sym, data in per_symbol.items():
        params_out[sym] = data.get("params", {})

    output = {
        "architecture": architecture,
        "per_symbol_params": params_out,
        "portfolio_stats": results.get("portfolio_stats", {}),
    }

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w") as f:
        json.dump(output, f, default=_default, indent=2)
    log(f"Parameters JSON saved: {path}")
    return str(path)
