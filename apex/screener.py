"""Strategy universe screener with per-symbol tuning and true holdout scoring."""
from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type
from urllib.request import Request, urlopen

import numpy as np
import optuna
import pandas as pd

from apex.ensemble.pnl import compute_pnl_stats
from apex.logging_util import log
from apex.strategies.base import StrategyBase


def load_sp500_symbols() -> List[str]:
    """Load current S&P 500 constituents from Wikipedia.

    Polygon uses dash class tickers for names such as BRK.B, so normalize dots
    to dashes.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise RuntimeError("Could not load S&P 500 table")
    symbols = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return sorted(symbols.str.upper().unique().tolist())


def _periods_per_year(cfg: Dict[str, Any]) -> int:
    tf = cfg.get("phase3_params", {}).get("exec_timeframe", "1H")
    bars = cfg.get("execution", {})
    if tf == "5min":
        return int(bars.get("bars_per_day_5min", 78)) * 252
    if tf == "1H":
        return int(bars.get("bars_per_day_1h", 7)) * 252
    return 252


def _strategy_stats(strategy: StrategyBase, df: pd.DataFrame,
                    periods_per_year: int, commission_pct: float) -> Dict[str, Any]:
    data = {
        "exec_df_1H": df.reset_index(drop=True),
        "regime_state": pd.Series(["R2"] * len(df)),
    }
    signals = strategy.compute_signals(data)
    positions = strategy.compute_position_size(data, signals)
    stats = compute_pnl_stats(
        positions,
        data["exec_df_1H"]["close"],
        periods_per_year=periods_per_year,
        commission_pct=commission_pct,
    )
    stats["entry_long"] = int(signals.get("entry_long", pd.Series()).sum())
    stats["entry_short"] = int(signals.get("entry_short", pd.Series()).sum())
    return stats


def _suggest_params(trial: optuna.Trial, tunable: Dict[str, tuple]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in tunable.items():
        ptype = spec[2] if len(spec) >= 3 else None
        if ptype == "float":
            params[name] = trial.suggest_float(name, float(spec[0]), float(spec[1]))
        elif ptype == "categorical":
            options = spec[3] if len(spec) >= 4 else [True, False]
            params[name] = trial.suggest_categorical(name, options)
        else:
            params[name] = trial.suggest_int(name, int(spec[0]), int(spec[1]))
    return params


def _fitness(stats: Dict[str, Any], min_trades: int) -> float:
    trades = int(stats.get("n_trades", 0))
    if trades < min_trades:
        return -999.0 + trades
    ret = float(stats.get("total_return_pct", 0.0))
    sharpe = float(stats.get("sharpe_annualized", 0.0))
    dd = abs(float(stats.get("max_dd_pct", 0.0)))
    calmar = float(stats.get("calmar", 0.0))
    churn_penalty = max(0, trades - 120) * 0.02
    return sharpe + 0.03 * ret + 0.15 * min(calmar, 5.0) - 0.02 * dd - churn_penalty


def tune_strategy_for_symbol(strategy_cls: Type[StrategyBase], df: pd.DataFrame,
                             cfg: Dict[str, Any], n_trials: int = 40,
                             seed: int = 42) -> Dict[str, Any]:
    """Tune one StrategyBase class on one symbol's tune window."""
    periods = _periods_per_year(cfg)
    commission_pct = float(cfg.get("screening", {}).get("commission_pct", 0.05))
    min_trades = int(cfg.get("screening", {}).get("min_tune_trades", 5))

    sample = strategy_cls()
    tunable = sample.get_tunable_params()
    if n_trials <= 0 or not tunable:
        stats = _strategy_stats(sample, df, periods, commission_pct)
        return {
            "params": {},
            "score": _fitness(stats, min_trades=min_trades),
            "stats": stats,
            "n_trials": 0,
        }

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, tunable)
        strategy = strategy_cls(params=params)
        stats = _strategy_stats(strategy, df, periods, commission_pct)
        return _fitness(stats, min_trades=min_trades)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params) if study.best_trial is not None else {}
    strategy = strategy_cls(params=best_params)
    stats = _strategy_stats(strategy, df, periods, commission_pct)
    return {
        "params": best_params,
        "score": float(study.best_value) if study.best_trial is not None else -999.0,
        "stats": stats,
        "n_trials": n_trials,
    }


def run_strategy_universe_screener(
    strategy_cls: Type[StrategyBase],
    data_dict: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    output_dir: Path,
    n_trials: int = 40,
    top_n: int = 25,
    seed: int = 42,
) -> Dict[str, Any]:
    """Rank symbols for a StrategyBase class with train-only tuning.

    The ranking score is fit on each symbol's tune window. Holdout metrics are
    reported but must not be used to retune the strategy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    periods = _periods_per_year(cfg)
    commission_pct = float(cfg.get("screening", {}).get("commission_pct", 0.05))
    min_holdout_trades = int(cfg.get("screening", {}).get("min_holdout_trades", 3))

    rows: List[Dict[str, Any]] = []
    for idx, (sym, sd) in enumerate(data_dict.items(), 1):
        tune_df = sd.get("exec_df")
        holdout_df = sd.get("exec_df_holdout")
        if tune_df is None or holdout_df is None or len(tune_df) < 200 or len(holdout_df) < 50:
            log(f"  [{idx}/{len(data_dict)}] {sym}: skip insufficient data", "WARN")
            continue

        log(f"  [{idx}/{len(data_dict)}] Screening {sym}...")
        try:
            tuned = tune_strategy_for_symbol(
                strategy_cls, tune_df, cfg, n_trials=n_trials, seed=seed,
            )
            strategy = strategy_cls(params=tuned["params"])
            holdout_stats = _strategy_stats(strategy, holdout_df, periods, commission_pct)
        except Exception as e:
            log(f"    {sym}: ERROR {e}", "WARN")
            continue

        tune_stats = tuned["stats"]
        holdout_trades = int(holdout_stats.get("n_trades", 0))
        holdout_pass = (
            holdout_trades >= min_holdout_trades
            and float(holdout_stats.get("total_return_pct", 0.0)) > 0.0
            and float(holdout_stats.get("sharpe_annualized", 0.0)) > 0.5
        )
        row = {
            "symbol": sym,
            "strategy": strategy_cls.name,
            "rank_score": float(tuned["score"]),
            "n_trials": int(tuned["n_trials"]),
            "tune_return_pct": float(tune_stats.get("total_return_pct", 0.0)),
            "tune_sharpe": float(tune_stats.get("sharpe_annualized", 0.0)),
            "tune_max_dd_pct": float(tune_stats.get("max_dd_pct", 0.0)),
            "tune_trades": int(tune_stats.get("n_trades", 0)),
            "holdout_return_pct": float(holdout_stats.get("total_return_pct", 0.0)),
            "holdout_sharpe": float(holdout_stats.get("sharpe_annualized", 0.0)),
            "holdout_max_dd_pct": float(holdout_stats.get("max_dd_pct", 0.0)),
            "holdout_trades": holdout_trades,
            "holdout_pass": holdout_pass,
            "params": tuned["params"],
        }
        rows.append(row)
        log(
            f"    {sym}: tune Ret={row['tune_return_pct']:.1f}% "
            f"Sh={row['tune_sharpe']:.2f}; holdout Ret={row['holdout_return_pct']:.1f}% "
            f"Sh={row['holdout_sharpe']:.2f} trades={holdout_trades} "
            f"{'PASS' if holdout_pass else 'FAIL'}"
        )

    rows.sort(key=lambda r: r["rank_score"], reverse=True)
    top_rows = rows[:top_n]

    csv_path = output_dir / "strategy_universe_screen.csv"
    cols = [
        "symbol", "strategy", "rank_score", "n_trials",
        "tune_return_pct", "tune_sharpe", "tune_max_dd_pct", "tune_trades",
        "holdout_return_pct", "holdout_sharpe", "holdout_max_dd_pct",
        "holdout_trades", "holdout_pass", "params",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["params"] = json.dumps(out["params"], sort_keys=True)
            writer.writerow(out)

    payload = {
        "strategy": strategy_cls.name,
        "n_symbols": len(rows),
        "n_trials_per_symbol": n_trials,
        "top_n": top_n,
        "rank_warning": (
            "Symbols are ranked by tune-window score. Holdout columns are "
            "diagnostic and must not be used for another optimization pass."
        ),
        "top": top_rows,
        "all": rows,
    }
    json_path = output_dir / "strategy_universe_screen.json"
    json_path.write_text(json.dumps(payload, indent=2))
    log(f"Universe screen CSV: {csv_path}")
    log(f"Universe screen JSON: {json_path}")
    return payload


def limit_symbols(symbols: Iterable[str], max_symbols: Optional[int]) -> List[str]:
    out = list(dict.fromkeys(symbols))
    if max_symbols and max_symbols > 0:
        return out[:max_symbols]
    return out
