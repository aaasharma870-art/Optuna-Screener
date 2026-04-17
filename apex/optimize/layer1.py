"""Layer 1: Architecture search via Optuna."""

import math

import numpy as np

try:
    import optuna
except ImportError:
    optuna = None

from apex.logging_util import log
from apex.engine.backtest import full_backtest, DEFAULT_PARAMS
from apex.util.concept_parser import INDICATOR_REGISTRY
from apex.util.checkpoints import save_checkpoint


def _compute_fitness(stats):
    """Fitness = PF * sqrt(trades) * (1 - max_dd/100). Penalises small samples."""
    pf = stats.get("pf", 0.0)
    trades = stats.get("trades", 0)
    max_dd = stats.get("max_dd_pct", 100.0)
    if trades < 5 or pf <= 0:
        return -999.0
    return pf * math.sqrt(trades) * (1.0 - max_dd / 100.0)


def _mini_monte_carlo(trade_pnls, n_sims=200, threshold=0.7):
    """
    Quick Monte Carlo: shuffle trades *n_sims* times, return a penalty
    multiplier in [0, 1] based on fraction of net-profitable runs.
    """
    if len(trade_pnls) < 5:
        return 0.0
    rng = np.random.RandomState(42)
    arr = np.array(trade_pnls)
    profit_count = 0
    for _ in range(n_sims):
        rng.shuffle(arr)
        equity = 10000.0
        for p in arr:
            equity *= (1.0 + p / 100.0)
        if equity > 10000.0:
            profit_count += 1
    prob_profit = profit_count / n_sims
    if prob_profit >= threshold:
        return 1.0
    return prob_profit / threshold


def _select_indicators_biased(trial, concept_bias, min_count=3, max_count=8):
    """Select indicator subset biased by concept weights."""
    all_names = list(INDICATOR_REGISTRY.keys())
    n_indicators = trial.suggest_int("n_indicators", min_count, min(max_count, len(all_names)))
    selected = []
    available = list(all_names)
    for idx in range(n_indicators):
        weights = np.array([concept_bias.get(nm, 1.0) for nm in available], dtype=np.float64)
        weights = np.clip(weights, 0.1, 10.0)
        weights /= weights.sum()
        cumulative = np.cumsum(weights)
        pick_val = trial.suggest_float(f"ind_pick_{idx}", 0.0, 1.0)
        pick_idx = int(np.searchsorted(cumulative, pick_val))
        pick_idx = min(pick_idx, len(available) - 1)
        selected.append(available[pick_idx])
        available.pop(pick_idx)
        if not available:
            break
    return list(set(selected)) if selected else all_names[:min_count]


def architecture_trial(trial, data_dict, concept_bias, cfg):
    """
    Optuna objective for Layer 1: search over architecture space.

    Runs a quick inner tune on a subset of symbols to evaluate architectural
    fitness.
    """
    indicators = _select_indicators_biased(trial, concept_bias)

    exit_combos = [
        ["fixed_target", "fixed_stop", "trailing_stop", "regime_exit", "time_exit"],
        ["trailing_stop", "regime_exit", "time_exit"],
        ["fixed_target", "fixed_stop", "regime_exit"],
        ["trailing_stop", "time_exit"],
        ["fixed_target", "trailing_stop", "regime_exit", "time_exit"],
    ]
    exit_idx = trial.suggest_int("exit_combo", 0, len(exit_combos) - 1)
    exit_methods = exit_combos[exit_idx]

    regime_model = trial.suggest_categorical("regime_model", ["ema", "volatility", "trend"])
    position_sizing = trial.suggest_categorical("position_sizing", ["equal", "volatility_scaled"])
    score_aggregation = trial.suggest_categorical("score_aggregation", ["additive", "weighted", "unanimous"])

    min_score = trial.suggest_int("min_score", max(2, len(indicators) // 2), max(3, len(indicators) - 1))

    architecture = {
        "indicators": indicators,
        "min_score": min_score,
        "exit_methods": exit_methods,
        "regime_model": regime_model,
        "position_sizing": position_sizing,
        "exec_timeframe": cfg.get("phase3_params", {}).get("exec_timeframe", "1H"),
        "score_aggregation": score_aggregation,
        "concept_weights": concept_bias,
    }

    inner_trials = cfg.get("optimization", {}).get("inner_trials", 30)
    symbols = list(data_dict.keys())[:5]
    if not symbols:
        return -999.0

    arch_fitness_values = []
    inner_study = None
    for sym in symbols:
        sym_data = data_dict[sym]
        df = sym_data.get("exec_df")
        daily_df = sym_data.get("daily_df")
        if df is None or len(df) < 100:
            continue

        def inner_objective(inner_trial):
            params = dict(DEFAULT_PARAMS)
            for ind_name in indicators:
                reg = INDICATOR_REGISTRY.get(ind_name, {})
                for pname, (lo, hi) in reg.get("params", {}).items():
                    if isinstance(lo, float) or isinstance(hi, float):
                        params[pname] = inner_trial.suggest_float(pname, float(lo), float(hi))
                    else:
                        params[pname] = inner_trial.suggest_int(pname, int(lo), int(hi))
            params["atr_stop_mult"] = inner_trial.suggest_float("atr_stop_mult", 0.8, 3.0)
            params["atr_target_mult"] = inner_trial.suggest_float("atr_target_mult", 1.5, 5.0)
            params["atr_trail_mult"] = inner_trial.suggest_float("atr_trail_mult", 0.5, 2.5)
            params["trail_activate_atr"] = inner_trial.suggest_float("trail_activate_atr", 0.5, 2.0)
            params["max_hold_bars"] = inner_trial.suggest_int("max_hold_bars", 10, 60)
            params["regime_bonus"] = inner_trial.suggest_int("regime_bonus", 0, 2)

            _, stats = full_backtest(df, daily_df, architecture, params)
            return _compute_fitness(stats)

        inner_study = optuna.create_study(direction="maximize",
                                          sampler=optuna.samplers.TPESampler(seed=42))
        inner_study.optimize(inner_objective, n_trials=inner_trials, show_progress_bar=False)
        best_val = inner_study.best_value if inner_study.best_trial else -999.0
        arch_fitness_values.append(best_val)

    if not arch_fitness_values or all(v <= -999 for v in arch_fitness_values):
        return -999.0

    valid = [v for v in arch_fitness_values if v > -999]
    if not valid:
        return -999.0

    mean_fitness = float(np.mean(valid))

    # Mini Monte Carlo penalty using best inner params on first symbol
    first_sym = symbols[0]
    first_data = data_dict[first_sym]
    df0 = first_data.get("exec_df")
    daily0 = first_data.get("daily_df")
    if df0 is not None and len(df0) >= 100 and inner_study is not None:
        best_params = dict(DEFAULT_PARAMS)
        if inner_study.best_trial:
            best_params.update(inner_study.best_params)
        trades0, _ = full_backtest(df0, daily0, architecture, best_params)
        pnls0 = [t["pnl_pct"] for t in trades0]
        mc_mult = _mini_monte_carlo(pnls0)
        mean_fitness *= mc_mult

    return mean_fitness


def layer1_architecture_search(data_dict, concept_bias, cfg):
    """
    Layer 1: Optuna search over the architecture space. Returns the best
    architecture dict.
    """
    arch_trials = cfg.get("optimization", {}).get("arch_trials", 20)
    log(f"=== LAYER 1: Architecture Search ({arch_trials} trials) ===")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))

    def objective(trial):
        return architecture_trial(trial, data_dict, concept_bias, cfg)

    study.optimize(objective, n_trials=arch_trials, show_progress_bar=True)

    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -9999, reverse=True)
    log("Top 5 architectures:")
    for rank, t in enumerate(sorted_trials[:5], 1):
        log(f"  #{rank}: fitness={t.value:.4f} params={t.params}")

    best = study.best_trial
    best_params = best.params

    # Reconstruct indicators from the best trial's biased picks
    n_ind = best_params.get("n_indicators", 5)
    all_names = list(INDICATOR_REGISTRY.keys())
    available = list(all_names)
    selected = []
    for idx in range(n_ind):
        pick_val = best_params.get(f"ind_pick_{idx}", 0.5)
        weights = np.array([concept_bias.get(nm, 1.0) for nm in available], dtype=np.float64)
        weights = np.clip(weights, 0.1, 10.0)
        weights /= weights.sum()
        cumulative = np.cumsum(weights)
        pick_idx = int(np.searchsorted(cumulative, pick_val))
        pick_idx = min(pick_idx, len(available) - 1)
        selected.append(available[pick_idx])
        available.pop(pick_idx)
        if not available:
            break
    selected = list(set(selected)) if selected else all_names[:3]

    exit_combos = [
        ["fixed_target", "fixed_stop", "trailing_stop", "regime_exit", "time_exit"],
        ["trailing_stop", "regime_exit", "time_exit"],
        ["fixed_target", "fixed_stop", "regime_exit"],
        ["trailing_stop", "time_exit"],
        ["fixed_target", "trailing_stop", "regime_exit", "time_exit"],
    ]
    exit_idx = best_params.get("exit_combo", 0)
    exit_methods = exit_combos[exit_idx]

    architecture = {
        "indicators": selected,
        "min_score": best_params.get("min_score", 4),
        "exit_methods": exit_methods,
        "regime_model": best_params.get("regime_model", "ema"),
        "position_sizing": best_params.get("position_sizing", "equal"),
        "exec_timeframe": cfg.get("phase3_params", {}).get("exec_timeframe", "1H"),
        "score_aggregation": best_params.get("score_aggregation", "additive"),
        "concept_weights": concept_bias,
    }

    log(f"Best architecture: {architecture}")
    save_checkpoint("layer1_architecture", {"architecture": architecture, "fitness": best.value})
    return architecture
