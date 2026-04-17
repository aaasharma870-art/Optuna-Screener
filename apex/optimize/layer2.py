"""Layer 2: Deep per-symbol parameter optimization."""

import numpy as np

try:
    import optuna
except ImportError:
    optuna = None

from apex.logging_util import log
from apex.engine.backtest import full_backtest, DEFAULT_PARAMS
from apex.optimize.layer1 import _compute_fitness
from apex.util.concept_parser import INDICATOR_REGISTRY
from apex.util.checkpoints import save_checkpoint


def deep_tune_objective(trial, sym, df_dict, architecture, cfg):
    """
    Per-symbol deep parameter tuning objective with walk-forward validation.

    Splits data 70/30 (IS/OOS), runs backtest on both, returns blended fitness.
    """
    sym_data = df_dict[sym]
    df = sym_data.get("exec_df")
    daily_df = sym_data.get("daily_df")

    if df is None or len(df) < 100:
        return -999.0

    # Suggest all numerical params
    params = dict(DEFAULT_PARAMS)
    active_indicators = architecture.get("indicators", [])
    for ind_name in active_indicators:
        reg = INDICATOR_REGISTRY.get(ind_name, {})
        for pname, (lo, hi) in reg.get("params", {}).items():
            if isinstance(lo, float) or isinstance(hi, float):
                params[pname] = trial.suggest_float(pname, float(lo), float(hi))
            else:
                params[pname] = trial.suggest_int(pname, int(lo), int(hi))

    params["atr_period"] = trial.suggest_int("atr_period", 10, 21)
    params["atr_stop_mult"] = trial.suggest_float("atr_stop_mult", 0.8, 3.0)
    params["atr_target_mult"] = trial.suggest_float("atr_target_mult", 1.5, 5.0)
    params["atr_trail_mult"] = trial.suggest_float("atr_trail_mult", 0.5, 2.5)
    params["trail_activate_atr"] = trial.suggest_float("trail_activate_atr", 0.3, 2.5)
    params["max_hold_bars"] = trial.suggest_int("max_hold_bars", 10, 60)
    params["regime_bonus"] = trial.suggest_int("regime_bonus", 0, 3)
    params["commission_pct"] = trial.suggest_float("commission_pct", 0.03, 0.10)
    params["min_score"] = trial.suggest_int("min_score_tune", 2, max(2, len(active_indicators)))
    architecture = dict(architecture)
    architecture["min_score"] = params["min_score"]

    # Walk-forward split: 70% IS, 30% OOS
    split_idx = int(len(df) * 0.7)
    df_is = df.iloc[:split_idx].reset_index(drop=True)
    df_oos = df.iloc[split_idx:].reset_index(drop=True)

    if daily_df is not None and len(daily_df) > 0:
        split_date = df["datetime"].iloc[split_idx]
        daily_is = daily_df[daily_df["datetime"] <= split_date].reset_index(drop=True)
        daily_oos = daily_df[daily_df["datetime"] > split_date].reset_index(drop=True)
        if len(daily_oos) < 20:
            daily_oos = daily_df.copy()
    else:
        daily_is = daily_df
        daily_oos = daily_df

    if len(df_is) < 80 or len(df_oos) < 30:
        return -999.0

    _, stats_is = full_backtest(df_is, daily_is, architecture, params)
    _, stats_oos = full_backtest(df_oos, daily_oos, architecture, params)

    fitness_is = _compute_fitness(stats_is)
    fitness_oos = _compute_fitness(stats_oos)

    if fitness_is <= -999 or fitness_oos <= -999:
        return -999.0

    # Blended fitness: favour OOS
    is_w = cfg.get("optimization", {}).get("fitness_is_weight", 0.4)
    oos_w = cfg.get("optimization", {}).get("fitness_oos_weight", 0.6)
    fitness = is_w * fitness_is + oos_w * fitness_oos

    # Require a minimum number of trades on both slices
    if stats_is.get("trades", 0) < 6 or stats_oos.get("trades", 0) < 3:
        return -999.0

    # Reject curve-fit PF artifacts
    if stats_is.get("pf", 0) > 12.0:
        return -999.0

    # Reject severe IS/OOS divergence (memorization signature)
    if abs(fitness_is) > 1e-6:
        divergence = abs(fitness_is - fitness_oos) / abs(fitness_is)
        if divergence > 0.8:
            return -999.0

    return fitness


def layer2_deep_tune(data_dict, architecture, survivors, cfg, basket=None):
    """
    Layer 2: per-symbol deep parameter optimization.

    Runs an Optuna study per symbol with TPE sampler.
    Returns dict of {sym: {"params": best_params, "stats": stats, ...}}.

    When *basket* is provided (dict[symbol] -> DataFrame from cross_asset.fetch_basket),
    each trade's pnl_pct is scaled by the basket alignment multiplier computed as of
    the trade's entry date.
    """
    deep_trials = cfg.get("optimization", {}).get("deep_trials", 100)
    log(f"=== LAYER 2: Deep Parameter Optimization ({deep_trials} trials/symbol) ===")

    # Cross-asset basket config
    basket_cfg = cfg.get("cross_asset_basket", {})
    if basket is not None:
        from apex.engine.portfolio import compute_basket_alignment
        short_days = basket_cfg.get("momentum_short_days", 21)
        long_days = basket_cfg.get("momentum_long_days", 63)
        align_thresh = basket_cfg.get("alignment_threshold", 3)
        size_mult = basket_cfg.get("size_multiplier", 1.25)
        log(f"  Basket alignment enabled: {list(basket.keys())}, "
            f"threshold={align_thresh}, mult={size_mult}")

    results = {}
    for idx, sym in enumerate(survivors, 1):
        if sym not in data_dict:
            log(f"  [{idx}/{len(survivors)}] {sym} - no data, skipping", "WARN")
            continue

        log(f"  [{idx}/{len(survivors)}] Tuning {sym}...")
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial, _sym=sym):
            return deep_tune_objective(trial, _sym, data_dict, architecture, cfg)

        study.optimize(objective, n_trials=deep_trials, show_progress_bar=False)

        if study.best_trial is None or study.best_value <= -999:
            log(f"    {sym}: no valid solution found", "WARN")
            continue

        best_params = dict(DEFAULT_PARAMS)
        best_params.update(study.best_params)

        sym_data = data_dict[sym]
        df = sym_data.get("exec_df")
        daily_df = sym_data.get("daily_df")
        trades, stats = full_backtest(df, daily_df, architecture, best_params)

        trade_pnls = [t["pnl_pct"] for t in trades]

        # Apply basket alignment scaling when basket is provided
        if basket is not None:
            scaled_pnls = []
            for t in trades:
                as_of = t.get("entry_datetime")
                if as_of is not None:
                    mult = compute_basket_alignment(
                        basket, as_of,
                        short_days=short_days,
                        long_days=long_days,
                        alignment_threshold=align_thresh,
                        size_multiplier=size_mult,
                    )
                else:
                    mult = 1.0
                scaled_pnls.append(t["pnl_pct"] * mult)
            trade_pnls = scaled_pnls

        results[sym] = {
            "params": best_params,
            "stats": stats,
            "trade_pnls": trade_pnls,
            "trades": trades,
            "fitness": study.best_value,
        }
        log(f"    {sym}: PF={stats['pf']:.2f}, WR={stats['wr_pct']:.1f}%, "
            f"trades={stats['trades']}, fitness={study.best_value:.4f}")

    save_checkpoint("layer2_tuned", {sym: {k: v for k, v in r.items() if k != "trades"}
                                     for sym, r in results.items()})
    log(f"Layer 2 complete: {len(results)} symbols tuned")
    return results
