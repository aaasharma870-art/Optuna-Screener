"""Layer 3: Robustness gauntlet (Monte Carlo, noise, regime stress, sensitivity)."""

import numpy as np

from apex.logging_util import log
from apex.engine.backtest import full_backtest
from apex.util.checkpoints import save_checkpoint


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


def noise_injection_test(df_dict, architecture, params, cfg):
    """
    Inject noise into data and measure backtest degradation.

      * Add +/-5% random noise to close prices
      * Shift close by 1 bar to simulate timing jitter

    Returns dict with clean_pf, noisy_pf, pf_retention.
    """
    df = df_dict.get("exec_df")
    daily_df = df_dict.get("daily_df")
    if df is None or len(df) < 100:
        return {"clean_pf": 0.0, "noisy_pf": 0.0, "pf_retention": 0.0}

    _, stats_clean = full_backtest(df, daily_df, architecture, params)
    clean_pf = stats_clean.get("pf", 0.0)

    rng = np.random.RandomState(123)
    df_noisy = df.copy()
    noise = rng.uniform(-0.05, 0.05, size=len(df_noisy))
    df_noisy["close"] = df_noisy["close"] * (1.0 + noise)
    df_noisy["high"] = np.maximum(df_noisy["high"], df_noisy["close"])
    df_noisy["low"] = np.minimum(df_noisy["low"], df_noisy["close"])
    df_noisy["close"] = df_noisy["close"].shift(1).bfill()

    _, stats_noisy = full_backtest(df_noisy, daily_df, architecture, params)
    noisy_pf = stats_noisy.get("pf", 0.0)

    pf_retention = noisy_pf / clean_pf if clean_pf > 0 else 0.0

    return {
        "clean_pf": round(clean_pf, 3),
        "noisy_pf": round(noisy_pf, 3),
        "pf_retention": round(pf_retention, 4),
    }


def regime_stress_test(df_dict, architecture, params, cfg):
    """
    Flip the regime model to measure regime sensitivity.

    The stressed run re-evaluates the strategy with an alternate regime model
    (``trend`` if the baseline was ``ema``, else ``ema``).
    """
    df = df_dict.get("exec_df")
    daily_df = df_dict.get("daily_df")
    if df is None or len(df) < 100:
        return {"normal_pf": 0.0, "stressed_pf": 0.0, "pf_retention": 0.0}

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


def param_sensitivity_test(df_dict, architecture, params, cfg):
    """
    Jitter each numerical parameter by +/-10% and measure PF stability.

    Returns dict of {param_name: {"stable": bool, "pf_range": [min, max]}}.
    """
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


def layer3_robustness_gauntlet(data_dict, architecture, tuned_results, cfg):
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

        noise = noise_injection_test(sym_data, architecture, params, cfg)
        noise_score = min(1.0, max(0.0, noise["pf_retention"]))

        stress = regime_stress_test(sym_data, architecture, params, cfg)
        stress_score = min(1.0, max(0.0, stress["pf_retention"]))

        sensitivity = param_sensitivity_test(sym_data, architecture, params, cfg)
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
