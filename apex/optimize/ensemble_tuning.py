"""Per-strategy Optuna tuning for the ensemble pipeline.

For each strategy in the ensemble, sweep its tunable params via Optuna
with CPCV median Sharpe as the objective. Returns the best params per
strategy, ready to be injected into the strategy instance before Layer
A/B/C run.
"""
from typing import Any, Dict, List

import numpy as np
import optuna
import pandas as pd

from apex.logging_util import log
from apex.validation.cpcv import cpcv_split


def _strategy_cpcv_sharpe(strategy, data: dict, n_blocks: int = 6,
                           n_test_blocks: int = 1, purge_bars: int = 10) -> float:
    """Score a strategy's current params via CPCV median Sharpe.

    Runs the strategy on each (train, test) fold's test slice, computes
    annualized Sharpe per fold, returns the median across folds.
    Returns -999 on insufficient data or all-fold failure.
    """
    df = data.get("exec_df_1H")
    if df is None or len(df) < 200:
        return -999.0
    n = len(df)

    sharpes = []
    for train_idx, test_idx in cpcv_split(n, n_blocks=n_blocks,
                                           n_test_blocks=n_test_blocks,
                                           purge_bars=purge_bars):
        if len(test_idx) < 50:
            continue
        # Build a sliced data dict with the test indices
        df_test = df.iloc[test_idx].reset_index(drop=True)
        regime_test = data.get("regime_state")
        if regime_test is not None:
            regime_test = regime_test.iloc[test_idx].reset_index(drop=True)
        sliced = dict(data)
        sliced["exec_df_1H"] = df_test
        sliced["regime_state"] = regime_test if regime_test is not None else pd.Series(["UNKNOWN"]*len(df_test))

        try:
            signals = strategy.compute_signals(sliced)
            positions = strategy.compute_position_size(sliced, signals)
            close = df_test["close"].values
            if len(close) < 2:
                continue
            price_returns = np.diff(close) / close[:-1]
            # Position from prior bar applies to current bar's return (shift-1, no look-ahead)
            pos_arr = positions.values[:-1] if len(positions) > 1 else positions.values
            strategy_returns = pos_arr * price_returns[:len(pos_arr)]
            if len(strategy_returns) < 2 or strategy_returns.std() < 1e-12:
                continue
            sharpe = float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(252))
            sharpes.append(sharpe)
        except Exception:
            continue

    if not sharpes:
        return -999.0
    return float(np.median(sharpes))


def tune_strategy(strategy_cls, data: dict, n_trials: int = 50,
                  n_blocks: int = 6, n_test_blocks: int = 1,
                  purge_bars: int = 10, seed: int = 42) -> Dict[str, Any]:
    """Run Optuna search over a strategy's tunable params, scoring by CPCV
    median Sharpe.

    Args:
        strategy_cls: the StrategyBase subclass (NOT an instance)
        data: per-symbol data dict (must contain exec_df_1H, regime_state, etc.)
        n_trials: Optuna trial count
        n_blocks, n_test_blocks, purge_bars: CPCV parameters

    Returns:
        {
            'best_params': dict[param_name -> value],
            'best_sharpe': float (CPCV median),
            'n_trials': int (actual count),
        }
    """
    # Build a sample instance to read tunable params
    sample = strategy_cls()
    tunable = sample.get_tunable_params()

    if not tunable:
        # No tunable params — return defaults
        return {"best_params": {}, "best_sharpe": _strategy_cpcv_sharpe(sample, data,
                                                                         n_blocks=n_blocks,
                                                                         n_test_blocks=n_test_blocks,
                                                                         purge_bars=purge_bars),
                "n_trials": 0}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _objective(trial):
        # Suggest each tunable param per its type
        params = {}
        for pname, spec in tunable.items():
            ptype = spec[2] if len(spec) >= 3 else None
            if ptype == "int":
                params[pname] = trial.suggest_int(pname, int(spec[0]), int(spec[1]))
            elif ptype == "float":
                params[pname] = trial.suggest_float(pname, float(spec[0]), float(spec[1]))
            elif ptype == "categorical":
                # categorical 4-tuple: (None, None, "categorical", [options])
                options = spec[3] if len(spec) >= 4 else [True, False]
                params[pname] = trial.suggest_categorical(pname, options)
            else:
                # default to int
                params[pname] = trial.suggest_int(pname, int(spec[0]), int(spec[1]))

        instance = strategy_cls(params=params)
        return _strategy_cpcv_sharpe(instance, data,
                                     n_blocks=n_blocks,
                                     n_test_blocks=n_test_blocks,
                                     purge_bars=purge_bars)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    if study.best_trial is None or study.best_value <= -999:
        log(f"  Tuning {strategy_cls.name}: no valid params found, using defaults", "WARN")
        return {"best_params": {}, "best_sharpe": -999.0, "n_trials": n_trials}

    return {
        "best_params": dict(study.best_params),
        "best_sharpe": float(study.best_value),
        "n_trials": n_trials,
    }


def tune_ensemble_strategies(strategy_classes: List, data_per_symbol: dict,
                              n_trials_per_strategy: int = 50) -> Dict[str, Dict[str, Any]]:
    """Tune every strategy in the ensemble against its primary symbol's data.

    Returns: {strategy_name: {best_params, best_sharpe, primary_symbol, n_trials}}
    """
    results = {}
    primary_symbol = list(data_per_symbol.keys())[0] if data_per_symbol else None
    if primary_symbol is None:
        return results

    primary_data = data_per_symbol[primary_symbol]

    for cls in strategy_classes:
        log(f"  Tuning ensemble strategy: {cls.name} on {primary_symbol}...")
        result = tune_strategy(cls, primary_data, n_trials=n_trials_per_strategy)
        result["primary_symbol"] = primary_symbol
        results[cls.name] = result
        log(f"    {cls.name}: best CPCV Sharpe={result['best_sharpe']:.3f} "
            f"({result['n_trials']} trials)")

    return results
