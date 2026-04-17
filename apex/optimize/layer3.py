"""Layer 3: Robustness gauntlet (Monte Carlo, noise, regime stress, sensitivity)."""

import numpy as np
import pandas as pd

from apex.logging_util import log
from apex.engine.backtest import full_backtest
from apex.engine.strategy_backtest import strategy_full_backtest
from apex.util.checkpoints import save_checkpoint
from apex.validation.synthetic_mc import synthetic_price_mc, passes_synthetic_gate
from apex.validation.dsr import deflated_sharpe_ratio
from apex.validation.pbo import probability_of_backtest_overfitting


def monte_carlo_validate(trade_pnls, n_sims=3000, initial_equity=10000):
    """
    Monte Carlo validation: shuffle trade returns *n_sims* times and compute
    probability and percentile statistics on the resulting equity curves.
    """
    if len(trade_pnls) < 5:
        return {
            "prob_profit": 0.0, "p5_equity": initial_equity,
            "p50_equity": initial_equity, "p95_equity": initial_equity,
            "p95_dd": 100.0,
        }

    rng = np.random.RandomState(42)
    arr = np.array(trade_pnls, dtype=np.float64)
    final_equities = []
    max_drawdowns = []

    for _ in range(n_sims):
        shuffled = rng.permutation(arr)
        equity = float(initial_equity)
        peak = equity
        worst_dd = 0.0
        for p in shuffled:
            equity *= (1.0 + p / 100.0)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
            if dd > worst_dd:
                worst_dd = dd
        final_equities.append(equity)
        max_drawdowns.append(worst_dd)

    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)

    return {
        "prob_profit": float(np.mean(final_equities > initial_equity)),
        "p5_equity": float(np.percentile(final_equities, 5)),
        "p50_equity": float(np.percentile(final_equities, 50)),
        "p95_equity": float(np.percentile(final_equities, 95)),
        "p95_dd": float(np.percentile(max_drawdowns, 95)),
    }


def noise_injection_test(df_dict, architecture, params, cfg, strategy_adapter=None):
    """
    Inject noise into data and measure backtest degradation.

    If strategy_adapter is provided, uses the user strategy's entry/exit logic
    instead of the pipeline's built-in backtest.
    """
    df = df_dict.get("exec_df")
    daily_df = df_dict.get("daily_df")
    if df is None or len(df) < 100:
        return {"clean_pf": 0.0, "noisy_pf": 0.0, "pf_retention": 0.0}

    sym = df_dict.get("_sym", "UNKNOWN")
    spy_df = df_dict.get("_spy_df")

    if strategy_adapter is not None:
        _, stats_clean = strategy_full_backtest(strategy_adapter, df, spy_df, sym)
    else:
        _, stats_clean = full_backtest(df, daily_df, architecture, params)
    clean_pf = stats_clean.get("pf", 0.0)

    rng = np.random.RandomState(123)
    df_noisy = df.copy()
    noise = rng.uniform(-0.05, 0.05, size=len(df_noisy))
    df_noisy["close"] = df_noisy["close"] * (1.0 + noise)
    df_noisy["high"] = np.maximum(df_noisy["high"], df_noisy["close"])
    df_noisy["low"] = np.minimum(df_noisy["low"], df_noisy["close"])
    df_noisy["close"] = df_noisy["close"].shift(1).bfill()

    if strategy_adapter is not None:
        _, stats_noisy = strategy_full_backtest(strategy_adapter, df_noisy, spy_df, sym)
    else:
        _, stats_noisy = full_backtest(df_noisy, daily_df, architecture, params)
    noisy_pf = stats_noisy.get("pf", 0.0)

    pf_retention = noisy_pf / clean_pf if clean_pf > 0 else 0.0

    return {
        "clean_pf": round(clean_pf, 3),
        "noisy_pf": round(noisy_pf, 3),
        "pf_retention": round(pf_retention, 4),
    }


def regime_stress_test(df_dict, architecture, params, cfg, strategy_adapter=None):
    """
    Flip the regime model to measure regime sensitivity.
    For strategy mode, re-run the user strategy on the same data (no regime to flip,
    so we measure consistency by running twice with a small price shift).
    """
    df = df_dict.get("exec_df")
    daily_df = df_dict.get("daily_df")
    if df is None or len(df) < 100:
        return {"normal_pf": 0.0, "stressed_pf": 0.0, "pf_retention": 0.0}

    sym = df_dict.get("_sym", "UNKNOWN")
    spy_df = df_dict.get("_spy_df")

    if strategy_adapter is not None:
        _, stats_normal = strategy_full_backtest(strategy_adapter, df, spy_df, sym)
        normal_pf = stats_normal.get("pf", 0.0)
        # Stress: shift prices by 1 bar (timing stress)
        df_stressed = df.copy()
        df_stressed["close"] = df_stressed["close"].shift(1).bfill()
        df_stressed["open"] = df_stressed["open"].shift(1).bfill()
        _, stats_stressed = strategy_full_backtest(strategy_adapter, df_stressed, spy_df, sym)
        stressed_pf = stats_stressed.get("pf", 0.0)
    else:
        _, stats_normal = full_backtest(df, daily_df, architecture, params)
        normal_pf = stats_normal.get("pf", 0.0)
        arch_stressed = dict(architecture)
        baseline = architecture.get("regime_model", "ema")
        arch_stressed["regime_model"] = "trend" if baseline != "trend" else "ema"
        _, stats_stressed = full_backtest(df, daily_df, arch_stressed, params)
        stressed_pf = stats_stressed.get("pf", 0.0)

    pf_retention = stressed_pf / normal_pf if normal_pf > 0 else 0.0

    return {
        "normal_pf": round(normal_pf, 3),
        "stressed_pf": round(stressed_pf, 3),
        "pf_retention": round(pf_retention, 4),
    }


def param_sensitivity_test(df_dict, architecture, params, cfg, strategy_adapter=None):
    """
    Jitter parameters by +/-10% and measure PF stability.
    In strategy mode with no tunable params, returns a neutral score.
    """
    if strategy_adapter is not None and not strategy_adapter.tunable_params:
        return {"_no_params": {"stable": True, "pf_range": [1.0, 1.0], "base_pf": 1.0}}

    df = df_dict.get("exec_df")
    daily_df = df_dict.get("daily_df")
    if df is None or len(df) < 100:
        return {}

    _, stats_base = full_backtest(df, daily_df, architecture, params)
    base_pf = stats_base.get("pf", 0.0)

    sensitivity = {}
    numeric_params = {k: v for k, v in params.items()
                      if isinstance(v, (int, float)) and k not in ("commission_pct",)}

    for pname, pval in numeric_params.items():
        if pval == 0:
            continue
        pf_values = []
        for jitter in [-0.10, 0.10]:
            test_params = dict(params)
            jittered = pval * (1.0 + jitter)
            if isinstance(pval, int):
                jittered = max(1, int(round(jittered)))
            else:
                jittered = round(jittered, 6)
            test_params[pname] = jittered
            _, test_stats = full_backtest(df, daily_df, architecture, test_params)
            pf_values.append(test_stats.get("pf", 0.0))

        pf_min = min(pf_values)
        pf_max = max(pf_values)
        stable = True
        if base_pf > 0:
            if pf_min < base_pf * 0.7 or pf_max > base_pf * 1.3:
                stable = False
        sensitivity[pname] = {
            "stable": stable,
            "pf_range": [round(pf_min, 3), round(pf_max, 3)],
            "base_pf": round(base_pf, 3),
        }

    return sensitivity


def layer3_robustness_gauntlet(data_dict, architecture, tuned_results, cfg, strategy_adapter=None):
    """
    Layer 3: comprehensive robustness testing.

    Monte Carlo, noise injection, regime stress, and parameter sensitivity
    tests are combined into a composite score per symbol.
    """
    threshold = cfg.get("optimization", {}).get("robustness_threshold", 0.4)
    log(f"=== LAYER 3: Robustness Gauntlet (threshold={threshold}) ===")

    validated = {}
    robustness_data = {}

    for idx, (sym, result) in enumerate(tuned_results.items(), 1):
        log(f"  [{idx}/{len(tuned_results)}] Testing {sym}...")
        params = result["params"]
        trade_pnls = result["trade_pnls"]
        sym_data = data_dict.get(sym, {})

        mc = monte_carlo_validate(trade_pnls, n_sims=cfg.get("robustness", {}).get("monte_carlo_sims", 3000))
        mc_score = mc["prob_profit"]

        noise = noise_injection_test(sym_data, architecture, params, cfg, strategy_adapter=strategy_adapter)
        noise_score = min(1.0, max(0.0, noise["pf_retention"]))

        stress = regime_stress_test(sym_data, architecture, params, cfg, strategy_adapter=strategy_adapter)
        stress_score = min(1.0, max(0.0, stress["pf_retention"]))

        sensitivity = param_sensitivity_test(sym_data, architecture, params, cfg, strategy_adapter=strategy_adapter)
        if sensitivity:
            stable_count = sum(1 for v in sensitivity.values() if v["stable"])
            sensitivity_score = stable_count / len(sensitivity)
        else:
            sensitivity_score = 0.5

        scores = [max(0.001, mc_score), max(0.001, noise_score),
                  max(0.001, stress_score), max(0.001, sensitivity_score)]
        composite = float(np.prod(scores) ** (1.0 / len(scores)))

        robustness_data[sym] = {
            "mc": mc,
            "noise": noise,
            "stress": stress,
            "sensitivity": sensitivity,
            "mc_score": round(mc_score, 4),
            "noise_score": round(noise_score, 4),
            "stress_score": round(stress_score, 4),
            "sensitivity_score": round(sensitivity_score, 4),
            "composite": round(composite, 4),
        }

        # --- Validation suite (config-gated) ---
        vcfg = cfg.get("validation", {})
        robust_score = composite

        # Synthetic MC gate
        smc_cfg = vcfg.get("synthetic_mc", {})
        if smc_cfg.get("enabled", False):
            exec_df = sym_data.get("exec_df")
            if exec_df is not None and "close" in exec_df.columns and len(exec_df) >= 20:
                close_series = exec_df["close"]
                smc_paths = synthetic_price_mc(
                    close_series,
                    n_paths=smc_cfg.get("n_paths", 1000),
                    block_size=smc_cfg.get("block_size", 5),
                )
                final_prices = smc_paths[:, -1]
                profitable = float(np.mean(final_prices > close_series.iloc[0]))
                min_pct = smc_cfg.get("min_profitable_pct", 20)
                smc_pass = passes_synthetic_gate(profitable, min_pass_pct=min_pct)
                robustness_data[sym]["synthetic_mc_profitable_frac"] = round(profitable, 4)
                robustness_data[sym]["synthetic_mc_pass"] = smc_pass
                if not smc_pass:
                    robust_score *= 0.5
                    log(f"    {sym}: Synthetic MC FAIL ({profitable*100:.1f}% < {min_pct}%) -> score * 0.5")

        # DSR computation
        dsr_cfg = vcfg.get("dsr", {})
        if dsr_cfg.get("enabled", False):
            obs_sharpe = result.get("stats", {}).get("sharpe", 0.0)
            n_trials_dsr = len(tuned_results)
            sr_vals = [r.get("stats", {}).get("sharpe", 0.0) for r in tuned_results.values()]
            sr_var = float(np.var(sr_vals)) if len(sr_vals) > 1 else 0.01
            skew_val = float(np.mean([(s - np.mean(sr_vals))**3 for s in sr_vals]) / max(np.std(sr_vals)**3, 1e-12)) if len(sr_vals) > 2 else 0.0
            n_samples = result.get("stats", {}).get("trades", 100)
            dsr_val = deflated_sharpe_ratio(
                observed_sr=obs_sharpe,
                n_trials=max(n_trials_dsr, 2),
                sr_variance=max(sr_var, 1e-6),
                skew=skew_val,
                kurtosis=3.0,
                n_samples=max(n_samples, 2),
            )
            robustness_data[sym]["dsr"] = round(dsr_val, 4)
            log(f"    {sym}: DSR = {dsr_val:.4f}")

        # PBO (needs IS/OOS matrix from Layer 2)
        pbo_cfg = vcfg.get("pbo", {})
        if pbo_cfg.get("enabled", False):
            is_oos = result.get("is_oos_matrix")
            if is_oos is not None:
                is_mat = np.array(is_oos.get("is_scores", []))
                oos_mat = np.array(is_oos.get("oos_scores", []))
                if is_mat.ndim == 2 and oos_mat.ndim == 2 and is_mat.shape[0] >= 2:
                    pbo_val = probability_of_backtest_overfitting(is_mat, oos_mat)
                    robustness_data[sym]["pbo"] = round(pbo_val, 4)
                    log(f"    {sym}: PBO = {pbo_val:.4f}")

        # Use adjusted score if validation penalised it
        composite = robust_score

        log(f"    {sym}: MC={mc_score:.3f} Noise={noise_score:.3f} "
            f"Stress={stress_score:.3f} Sens={sensitivity_score:.3f} "
            f"Composite={composite:.3f}")

        if composite >= threshold:
            validated[sym] = result
            validated[sym]["robustness"] = robustness_data[sym]
            log(f"    {sym}: PASSED")
        else:
            log(f"    {sym}: REJECTED (composite {composite:.3f} < {threshold})")

    save_checkpoint("layer3_robustness", robustness_data)
    log(f"Layer 3 complete: {len(validated)}/{len(tuned_results)} passed")
    return validated, robustness_data
