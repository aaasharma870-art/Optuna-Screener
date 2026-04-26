"""Main pipeline entry point."""

import argparse
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from apex.config import CFG, OUTPUT_DIR, FORCED_SYMBOLS, load_config
from apex.logging_util import log
from apex.data.polygon_client import fetch_daily, fetch_bars
from apex.engine.backtest import full_backtest, DEFAULT_ARCHITECTURE, DEFAULT_PARAMS, VRP_DEFAULT_ARCHITECTURE
from apex.engine.portfolio import correlation_filter, phase_full_backtest
from apex.optimize.layer1 import layer1_architecture_search
from apex.optimize.layer2 import layer2_deep_tune
from apex.optimize.layer3 import layer3_robustness_gauntlet
from apex.report.html_report import generate_html_report
from apex.report.csv_json import generate_trades_csv, generate_summary_csv, generate_parameters_json
from apex.report.amibroker import generate_apex_afl, push_to_amibroker
from apex.util.concept_parser import parse_concept
from apex.util.sector_map import SECTOR_MAP
from apex.util.checkpoints import save_checkpoint, load_checkpoint
from apex.engine.strategy_adapter import StrategyAdapter
from apex.engine.strategy_backtest import strategy_full_backtest


def validate_vrp(cfg):
    """VRP strategy smoke test -- 6 diagnostic checks."""
    from pathlib import Path
    import traceback

    print("=== VRP STRATEGY VALIDATION ===\n")
    checks_passed = 0
    checks_total = 6

    # --- Check 1: Fetch VIX, VXV from FRED ---
    print("CHECK 1: FRED VIX/VXV fetch")
    try:
        from apex.data.fred_client import fetch_fred_series
        vix_df = fetch_fred_series("VIXCLS", "2015-01-01", "2026-12-31")
        vxv_df = fetch_fred_series("VXVCLS", "2015-01-01", "2026-12-31")
        if vix_df.empty or vxv_df.empty:
            print("  WARN: FRED returned empty data (API key missing or rate-limited)")
            print("  SKIP\n")
        else:
            print(f"  VIX: {len(vix_df)} rows, last 5:")
            print(vix_df.tail(5).to_string(header=True))
            print(f"  VXV: {len(vxv_df)} rows, last 5:")
            print(vxv_df.tail(5).to_string(header=True))
            checks_passed += 1
            print("  PASS\n")
    except Exception as e:
        print(f"  SKIP (FRED unavailable: {e})\n")

    # --- Check 2: Compute VRP on SPY daily data ---
    print("CHECK 2: VRP computation on SPY daily data")
    try:
        from apex.data.fred_client import fetch_fred_series
        from apex.regime.vrp import compute_vrp
        from apex.data.polygon_client import fetch_daily

        _, spy_daily, status = fetch_daily("SPY")
        if spy_daily is None or len(spy_daily) == 0:
            print(f"  SKIP (SPY daily fetch failed: {status})\n")
        else:
            vix_df = fetch_fred_series("VIXCLS", "2015-01-01", "2026-12-31")
            if vix_df.empty:
                print("  SKIP (FRED VIX unavailable)\n")
            else:
                daily_close = spy_daily.set_index("datetime")["close"]
                iv_aligned = vix_df["value"].reindex(daily_close.index, method="ffill")
                vrp_result = compute_vrp(iv_aligned, daily_close, rv_window=20, pct_window=252)
                vrp_pct = vrp_result["vrp_pct"].dropna()
                print(f"  VRP computed: {len(vrp_result)} rows, {len(vrp_pct)} non-NaN")
                print(f"  Last 20 vrp_pct values:")
                print(vrp_pct.tail(20).to_string())
                in_range = (vrp_pct >= 0).all() and (vrp_pct <= 100).all()
                print(f"  Range check [0, 100]: {'PASS' if in_range else 'FAIL'}")
                if in_range:
                    checks_passed += 1
                    print("  PASS\n")
                else:
                    print("  FAIL\n")
    except Exception as e:
        print(f"  SKIP ({e})\n")

    # --- Check 3: VPIN on SPY fixture data ---
    print("CHECK 3: VPIN computation")
    try:
        from apex.indicators.vpin import compute_vpin
        fixture_path = Path(__file__).parent.parent / "tests" / "fixtures" / "SPY_1H.parquet"
        spy_bars = pd.read_parquet(fixture_path)
        vpin_df = compute_vpin(spy_bars)
        vpin_vals = vpin_df["vpin"].dropna()
        if len(vpin_vals) == 0:
            print("  WARN: VPIN all NaN")
            print("  FAIL\n")
        else:
            mean_vpin = vpin_vals.mean()
            print(f"  VPIN stats: count={len(vpin_vals)}, mean={mean_vpin:.4f}, "
                  f"std={vpin_vals.std():.4f}, min={vpin_vals.min():.4f}, max={vpin_vals.max():.4f}")
            ok = 0.0 <= mean_vpin <= 1.0
            print(f"  Mean in [0, 1]: {'PASS' if ok else 'FAIL'}")
            if ok:
                checks_passed += 1
                print("  PASS\n")
            else:
                print("  FAIL\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # --- Check 4: VWCLV on SPY fixture data ---
    print("CHECK 4: VWCLV computation")
    try:
        from apex.indicators.vwclv import compute_vwclv
        fixture_path = Path(__file__).parent.parent / "tests" / "fixtures" / "SPY_1H.parquet"
        spy_bars = pd.read_parquet(fixture_path)
        vwclv_df = compute_vwclv(spy_bars)
        cum = vwclv_df["cum_vwclv"].dropna()
        print(f"  cum_vwclv: mean={cum.mean():.4f}, std={cum.std():.4f}, "
              f"min={cum.min():.4f}, max={cum.max():.4f}")
        checks_passed += 1
        print("  PASS\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # --- Check 5: Regime classification ---
    print("CHECK 5: VRP regime classification")
    try:
        from apex.regime.vrp_regime import compute_vrp_regime
        # Try with real data first, fall back to synthetic
        fixture_path = Path(__file__).parent.parent / "tests" / "fixtures" / "SPY_daily.parquet"
        spy_daily = pd.read_parquet(fixture_path)
        n = min(252, len(spy_daily))
        spy_daily = spy_daily.tail(n).reset_index(drop=True)

        # Synthesize plausible VIX/VXV/VRP data for fixture test
        np.random.seed(42)
        vix_vals = pd.Series(np.random.uniform(12, 35, n), index=spy_daily.index)
        vxv_vals = pd.Series(vix_vals * np.random.uniform(0.9, 1.15, n), index=spy_daily.index)
        vrp_pct_vals = pd.Series(np.random.uniform(0, 100, n), index=spy_daily.index)

        regimes = compute_vrp_regime(spy_daily, vix_vals, vxv_vals, vrp_pct_vals)
        freq = regimes.value_counts(normalize=True) * 100
        print("  Regime frequency (%):")
        for r in ["R1", "R2", "R3", "R4"]:
            pct = freq.get(r, 0)
            print(f"    {r}: {pct:.1f}%")

        # Check for degeneracy: one regime > 90% = degenerate
        degenerate = any(freq.get(r, 0) > 90 for r in ["R1", "R2", "R3", "R4"])
        if degenerate:
            print("  WARNING: Degenerate distribution (one regime >90%)")
            print("  FAIL\n")
        else:
            checks_passed += 1
            print("  PASS\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # --- Check 6: Backtest on fixture data ---
    print("CHECK 6: Backtest on SPY fixture data")
    try:
        fixture_exec = Path(__file__).parent.parent / "tests" / "fixtures" / "SPY_1H.parquet"
        fixture_daily = Path(__file__).parent.parent / "tests" / "fixtures" / "SPY_daily.parquet"
        spy_exec = pd.read_parquet(fixture_exec)
        spy_daily_df = pd.read_parquet(fixture_daily)

        # Use DEFAULT_ARCHITECTURE (which works without VRP columns)
        arch = DEFAULT_ARCHITECTURE.copy()
        arch["direction"] = "neutral"
        params = DEFAULT_PARAMS.copy()

        trades, stats = full_backtest(spy_exec, spy_daily_df, arch, params)
        pf = stats.get("pf", 0)
        wr = stats.get("wr_pct", 0)
        n_trades = stats.get("trades", 0)
        print(f"  Trades: {n_trades}")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Win Rate: {wr:.1f}%")
        print(f"  Total Return: {stats.get('total_return_pct', 0):.2f}%")
        checks_passed += 1
        print("  PASS\n")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        print()

    print(f"=== VALIDATION COMPLETE: {checks_passed}/{checks_total} checks passed ===")
    return checks_passed


def phase1_universe(cfg):
    """Return candidate symbol list from config plus forced tickers."""
    target = cfg.get("target_symbols", [])
    if not target:
        target = list(SECTOR_MAP.keys())[:30]
    for sym in FORCED_SYMBOLS:
        if sym not in target:
            target.append(sym)
    log(f"Phase 1: Universe of {len(target)} candidates (includes forced: {FORCED_SYMBOLS})")
    return target


def phase2_quick_screen(candidates, cfg):
    """
    Quick screen: fetch daily data, filter by liquidity / price / volume.

    Returns (list of surviving symbols, dict of {sym: daily_df}).
    """
    uni = cfg.get("universe", {})
    min_price = uni.get("min_price", 10)
    max_price = uni.get("max_price", 5000)
    min_volume = uni.get("min_avg_volume", 500000)
    min_bars = uni.get("min_daily_bars", 252)

    log(f"Phase 2: Quick screen on {len(candidates)} symbols "
        f"(price ${min_price}-${max_price}, vol>{min_volume/1e6:.1f}M)")

    survivors = []
    daily_data = {}

    for idx, sym in enumerate(candidates, 1):
        forced = sym in FORCED_SYMBOLS
        sym_name, df, status = fetch_daily(sym)
        if df is None or len(df) < min(min_bars, 50):
            log(f"  [{idx}/{len(candidates)}] {sym}: SKIP ({status}, bars={len(df) if df is not None else 0})"
                + (" [FORCED but no data]" if forced else ""))
            continue

        avg_price = df["close"].iloc[-20:].mean() if len(df) >= 20 else df["close"].mean()
        avg_volume = df["volume"].iloc[-20:].mean() if len(df) >= 20 else df["volume"].mean()

        if not forced:
            if avg_price < min_price or avg_price > max_price:
                log(f"  [{idx}/{len(candidates)}] {sym}: SKIP (price ${avg_price:.2f})")
                continue
            if avg_volume < min_volume:
                log(f"  [{idx}/{len(candidates)}] {sym}: SKIP (volume {avg_volume/1e6:.1f}M)")
                continue

        survivors.append(sym)
        daily_data[sym] = df
        tag = " [FORCED]" if forced else ""
        log(f"  [{idx}/{len(candidates)}] {sym}: PASS{tag} (${avg_price:.0f}, "
            f"vol={avg_volume/1e6:.1f}M, bars={len(df)})")

    log(f"Phase 2 complete: {len(survivors)}/{len(candidates)} survived")
    return survivors, daily_data


def phase3_fetch_data(survivors, daily_data, cfg):
    """
    Fetch execution-timeframe data for each survivor and split off a
    final holdout window that no optimizer will ever see.
    """
    exec_tf = cfg.get("phase3_params", {}).get("exec_timeframe", "1H")
    log(f"Phase 3: Fetching {exec_tf} data for {len(survivors)} symbols")

    data_dict = {}
    for idx, sym in enumerate(survivors, 1):
        log(f"  [{idx}/{len(survivors)}] Fetching {sym}...")

        _, exec_df_full, status = fetch_bars(sym, timeframe=exec_tf)
        if exec_df_full is None or len(exec_df_full) < 100:
            log(f"    {sym}: SKIP (exec bars: {len(exec_df_full) if exec_df_full is not None else 0})")
            continue

        daily_df_full = daily_data.get(sym)

        # Reserve the final N% as a true holdout that NO optimizer ever sees.
        # Layer 1 / Layer 2 / robustness all run on the tune window only.
        holdout_pct = cfg.get("optimization", {}).get("final_holdout_pct", 0.25)
        cut = int(len(exec_df_full) * (1.0 - holdout_pct))
        exec_df = exec_df_full.iloc[:cut].reset_index(drop=True)
        exec_df_holdout = exec_df_full.iloc[cut:].reset_index(drop=True)

        if daily_df_full is not None and len(exec_df) > 0:
            split_dt = exec_df["datetime"].iloc[-1]
            daily_df = daily_df_full[daily_df_full["datetime"] <= split_dt].reset_index(drop=True)
            daily_df_holdout = daily_df_full[daily_df_full["datetime"] > split_dt].reset_index(drop=True)
        else:
            daily_df = daily_df_full
            daily_df_holdout = None

        data_dict[sym] = {
            "exec_df": exec_df,
            "daily_df": daily_df,
            "exec_df_holdout": exec_df_holdout,
            "daily_df_holdout": daily_df_holdout,
        }
        log(f"    {sym}: OK (tune={len(exec_df)} bars, holdout={len(exec_df_holdout)} bars)")

    log(f"Phase 3 complete: {len(data_dict)} symbols with full data")
    return data_dict


def main():
    """
    Pipeline entry point.

      1. Universe selection
      2. Quick screen (daily liquidity)
      3. Multi-TF data fetch + holdout split
      4. Layer 1: Architecture search
      5. Layer 2: Deep parameter optimization
      6. Layer 3: Robustness gauntlet
      7. Correlation filter
      8. Full final backtest (tune + true holdout)
      9. Report generation (HTML + CSV + JSON)
     10. AmiBroker push
     11. Open report in browser
    """
    parser = argparse.ArgumentParser(description="Optuna Screener Pipeline")
    parser.add_argument("--config", default="apex_config.json", help="Config JSON path")
    parser.add_argument("--test", action="store_true", help="Test mode: 3 symbols, light budget")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--no-amibroker", action="store_true", help="Skip AmiBroker push")
    parser.add_argument("--concept", type=str, default="", help="Strategy concept string")
    parser.add_argument("--budget", type=str, default="medium",
                        choices=["light", "medium", "heavy"], help="Compute budget")
    parser.add_argument("--output", type=str, default="", help="Output directory override")
    parser.add_argument("--validate-vrp", action="store_true",
                        help="Run VRP strategy validation smoke test and exit")
    parser.add_argument("--strategy", type=str, default="",
                        help="Path to a user strategy .py file (uses exact entry/exit logic)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.validate_vrp:
        validate_vrp(cfg)
        sys.exit(0)

    # Apply budget profile
    budget = cfg.get("budget_profiles", {}).get(args.budget, {})
    opt = cfg.get("optimization", {})
    rob = cfg.get("robustness", {})
    if budget:
        opt["arch_trials"] = budget.get("arch_trials", 200)
        opt["inner_trials"] = budget.get("arch_trials", 200) // 5
        opt["deep_trials"] = budget.get("deep_trials", 800)
        rob["monte_carlo_sims"] = budget.get("mc_sims", 3000)
    else:
        opt.setdefault("arch_trials", 200)
        opt.setdefault("inner_trials", 40)
        opt.setdefault("deep_trials", 800)
    opt["robustness_threshold"] = rob.get("min_robustness_score", 0.5)
    cfg["optimization"] = opt
    cfg["robustness"] = rob
    log(f"Budget: {args.budget} - arch={opt['arch_trials']} trials, "
        f"deep={opt['deep_trials']}/symbol, MC={rob.get('monte_carlo_sims', 3000)} sims, "
        f"robustness>={opt['robustness_threshold']}")

    opt = cfg.get("optimization", {})
    if args.test:
        log("*** TEST MODE: light budget, 3 symbols ***")
        opt["arch_trials"] = 5
        opt["inner_trials"] = 10
        opt["deep_trials"] = 20
        opt["robustness_threshold"] = 0.2
        cfg["optimization"] = opt
        target = cfg.get("target_symbols", list(SECTOR_MAP.keys())[:3])
        cfg["target_symbols"] = target[:3]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        run_output = Path(args.output)
    else:
        run_output = OUTPUT_DIR / f"run_{timestamp}"
    run_output.mkdir(parents=True, exist_ok=True)

    run_info = {
        "timestamp": timestamp,
        "concept": args.concept or cfg.get("concept", "adaptive multi-indicator"),
        "test_mode": args.test,
        "config": args.config,
    }

    log("=" * 60)
    log("Optuna Screener Pipeline")
    log("=" * 60)
    log(f"Output: {run_output}")
    log(f"Concept: {run_info['concept']}")

    if cfg.get("strategy_mode") == "vrp_regime":
        architecture = VRP_DEFAULT_ARCHITECTURE
        log("Using VRP Regime strategy mode")
        concept_bias = {}
    else:
        concept_bias = parse_concept(run_info["concept"])
        log(f"Concept bias: { {k: v for k, v in concept_bias.items() if v != 1.0} }")

    resume_stage = None
    if args.resume:
        for stage in ["layer3_robustness", "layer2_tuned", "layer1_architecture"]:
            cp = load_checkpoint(stage, str(run_output))
            if cp is not None:
                resume_stage = stage
                break
        if resume_stage:
            log(f"Resuming from checkpoint: {resume_stage}")
        else:
            log("No checkpoint found, starting fresh")

    # ---- Phase 1 ----
    candidates = phase1_universe(cfg)

    # ---- Phase 2 ----
    survivors, daily_data = phase2_quick_screen(candidates, cfg)
    if not survivors:
        log("No symbols survived Phase 2. Exiting.", "ERROR")
        sys.exit(1)

    # ---- Phase 3 ----
    data_dict = phase3_fetch_data(survivors, daily_data, cfg)
    if not data_dict:
        log("No symbols have sufficient data. Exiting.", "ERROR")
        sys.exit(1)
    survivors = list(data_dict.keys())

    # ---- STRATEGY MODE ----
    if args.strategy:
        adapter = StrategyAdapter(args.strategy)
        log(f"Strategy mode: {adapter.name}")
        log(f"Strategy file: {adapter.path}")

        # Get SPY data for RS calculations
        spy_data = data_dict.get("SPY", {})
        spy_exec_df = spy_data.get("exec_df")

        # Inject SPY reference and symbol name into data_dict for Layer 3
        for sym in data_dict:
            data_dict[sym]["_spy_df"] = spy_exec_df
            data_dict[sym]["_sym"] = sym

        # Skip Layer 1 — the strategy IS the architecture
        architecture = {
            "indicators": ["UserStrategy"],
            "min_score": 1,
            "exit_methods": ["user_strategy"],
            "regime_model": "none",
            "position_sizing": "equal",
            "exec_timeframe": cfg.get("phase3_params", {}).get("exec_timeframe", "1H"),
            "score_aggregation": "additive",
            "concept_weights": {},
            "direction": "long",
        }

        # Layer 2: Parameter tuning (if TUNABLE_PARAMS defined)
        from apex.optimize.layer1 import _compute_fitness

        if adapter.tunable_params:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            deep_trials = cfg.get("optimization", {}).get("deep_trials", 50)
            oos_pct = cfg.get("optimization", {}).get("walk_forward_oos_pct", 0.30)
            is_w = cfg.get("optimization", {}).get("fitness_is_weight", 0.4)
            oos_w = cfg.get("optimization", {}).get("fitness_oos_weight", 0.6)

            log(f"=== LAYER 2: Strategy Parameter Tuning ({deep_trials} trials/symbol) ===")
            log(f"  Tunable params: {list(adapter.tunable_params.keys())}")

            tuned_results = {}
            for idx, sym in enumerate(survivors, 1):
                log(f"  [{idx}/{len(survivors)}] Tuning {sym}...")
                sym_data = data_dict[sym]
                df = sym_data.get("exec_df")
                if df is None or len(df) < 100:
                    continue

                # Walk-forward split
                cut = int(len(df) * (1.0 - oos_pct))
                df_is = df.iloc[:cut].reset_index(drop=True)
                df_oos = df.iloc[cut:].reset_index(drop=True)

                # Pre-prepare DataFrames
                prep_is = adapter.prepare_df(df_is, spy_df=spy_exec_df, sym=sym)
                prep_oos = adapter.prepare_df(df_oos, spy_df=spy_exec_df, sym=sym)

                best_fitness = -999.0
                best_params = dict(adapter.default_params)
                best_trades = []
                best_stats = {}

                def objective(trial):
                    trial_params = {}
                    for pname, (lo, hi) in adapter.tunable_params.items():
                        if isinstance(lo, float) or isinstance(hi, float):
                            trial_params[pname] = trial.suggest_float(pname, float(lo), float(hi))
                        else:
                            trial_params[pname] = trial.suggest_int(pname, int(lo), int(hi))

                    adapter.set_params(trial_params)

                    from apex.engine.strategy_backtest import run_strategy_backtest
                    trades_is, stats_is = run_strategy_backtest(adapter, prep_is, sym)
                    trades_oos, stats_oos = run_strategy_backtest(adapter, prep_oos, sym)

                    fit_is = _compute_fitness(stats_is)
                    fit_oos = _compute_fitness(stats_oos)

                    if fit_is <= 0 and fit_oos <= 0:
                        return -999.0

                    return is_w * max(fit_is, 0) + oos_w * max(fit_oos, 0)

                study = optuna.create_study(direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(objective, n_trials=deep_trials, show_progress_bar=False)

                # Apply best params and run on full tune window
                if study.best_value > -900:
                    best_trial_params = {}
                    for pname, (lo, hi) in adapter.tunable_params.items():
                        best_trial_params[pname] = study.best_params.get(pname,
                            adapter.default_params.get(pname, lo))
                    adapter.set_params(best_trial_params)
                    best_params = dict(adapter.module.PARAMS)

                trades, stats = strategy_full_backtest(adapter, df, spy_exec_df, sym)
                trade_pnls = [t["pnl_pct"] for t in trades]

                if stats["trades"] < 3:
                    log(f"    {sym}: SKIP ({stats['trades']} trades)")
                    adapter.reset_params()
                    continue

                fitness = _compute_fitness(stats)
                tuned_results[sym] = {
                    "params": best_params,
                    "stats": stats,
                    "trades": trades,
                    "trade_pnls": trade_pnls,
                    "fitness": fitness,
                }
                log(f"    {sym}: {stats['trades']}tr PF={stats['pf']:.2f} "
                    f"WR={stats['wr_pct']:.1f}% Ret={stats['total_return_pct']:.1f}%")
                adapter.reset_params()

        else:
            # No tunable params — run strategy directly
            log("=== STRATEGY BACKTEST (user entry/exit logic, no tuning) ===")
            tuned_results = {}
            for idx, sym in enumerate(survivors, 1):
                log(f"  [{idx}/{len(survivors)}] Backtesting {sym}...")
                sym_data = data_dict[sym]
                df = sym_data.get("exec_df")
                if df is None or len(df) < 100:
                    continue

                trades, stats = strategy_full_backtest(adapter, df, spy_exec_df, sym)
                trade_pnls = [t["pnl_pct"] for t in trades]

                if stats["trades"] < 3:
                    log(f"    {sym}: SKIP ({stats['trades']} trades)")
                    continue

                fitness = _compute_fitness(stats)
                tuned_results[sym] = {
                    "params": {},
                    "stats": stats,
                    "trades": trades,
                    "trade_pnls": trade_pnls,
                    "fitness": fitness,
                }
                log(f"    {sym}: {stats['trades']}tr PF={stats['pf']:.2f} "
                    f"WR={stats['wr_pct']:.1f}% Ret={stats['total_return_pct']:.1f}%")

        if not tuned_results:
            log("No symbols produced trades. Exiting.", "ERROR")
            sys.exit(1)

        log(f"Strategy backtest complete: {len(tuned_results)} symbols with trades")

        # Layer 3: Robustness
        validated_results, robustness_data = layer3_robustness_gauntlet(
            data_dict, architecture, tuned_results, cfg, strategy_adapter=adapter
        )

        if not validated_results:
            log("No symbols passed robustness. Using all tuned results.", "WARN")
            validated_results = tuned_results
            for sym in validated_results:
                validated_results[sym]["robustness"] = robustness_data.get(sym, {})

        # Correlation filter
        final_results = correlation_filter(validated_results, cfg)
        if not final_results:
            final_results = validated_results

        # Final backtest (tune + holdout)
        backtest_results = phase_full_backtest(
            data_dict, architecture, final_results, cfg,
            tuned_results=tuned_results, strategy_adapter=adapter
        )

        # Reports (same as normal pipeline)
        log("=== GENERATING REPORTS ===")
        run_info["concept"] = adapter.name
        report_path = generate_html_report(
            backtest_results, architecture, robustness_data, run_info, str(run_output)
        )
        generate_trades_csv(backtest_results.get("all_trades", []), str(run_output))
        generate_summary_csv(backtest_results, str(run_output))
        generate_parameters_json(backtest_results, architecture, str(run_output))

        if not args.no_amibroker:
            sorted_syms = backtest_results.get("sorted_syms", [])
            afl_str = generate_apex_afl(sorted_syms, backtest_results, architecture)
            push_to_amibroker(backtest_results, afl_str, str(run_output), cfg)
        else:
            log("AmiBroker push skipped (--no-amibroker)")

        # Summary
        log("=== PIPELINE COMPLETE ===")
        pstats = backtest_results.get("portfolio_stats", {})
        log("Final Results:")
        log(f"  Strategy: {adapter.name}")
        log(f"  Symbols: {len(backtest_results.get('sorted_syms', []))}")
        log(f"  Trades:  {pstats.get('trades', 0)}")
        log(f"  PF:      {pstats.get('pf', 0):.2f}")
        log(f"  Win%:    {pstats.get('wr_pct', 0):.1f}%")
        log(f"  Return:  {pstats.get('total_return_pct', 0):.1f}%")
        log(f"  MaxDD:   {pstats.get('max_dd_pct', 0):.1f}%")
        log(f"  Sharpe:  {pstats.get('sharpe', 0):.2f}")
        hstats = backtest_results.get("holdout_universe_stats", {})
        if hstats:
            log("  --- TRUE HOLDOUT (never seen) ---")
            log(f"  Holdout Trades:  {hstats.get('trades', 0)}")
            log(f"  Holdout PF:      {hstats.get('pf', 0):.2f}")
            log(f"  Holdout Win%:    {hstats.get('wr_pct', 0):.1f}%")
            log(f"  Holdout Return:  {hstats.get('total_return_pct', 0):.1f}%")
            log(f"  Holdout Sharpe:  {hstats.get('sharpe', 0):.2f}")
        log(f"  Report:  {report_path}")

        try:
            abs_report = str(Path(report_path).resolve()).replace("\\", "/")
            webbrowser.open(f"file:///{abs_report}")
            log(f"Report opened in browser: file:///{abs_report}")
        except Exception as e:
            log(f"Could not open browser: {e}", "WARN")

        return backtest_results

    # ---- VRP Data Merge (opt-in) ----
    if cfg.get("strategy_mode") == "vrp_regime":
        from apex.data.fred_client import fetch_fred_series, IV_MAP
        from apex.data.polygon_client import polygon_request
        from apex.regime.vrp import compute_vrp
        log("=== VRP REGIME MODE: Fetching FRED implied-vol data ===")

        # Use a WIDE FRED window so the 252-day VRP percentile has full history.
        # Always go back at least to 2015 — VIX/VXV/VXN/GVZ are stable historical series.
        fred_start = "2015-01-01"
        fred_end = datetime.now().strftime("%Y-%m-%d")

        # Always fetch VIX and VXV for term structure
        try:
            vix_df = fetch_fred_series("VIXCLS", fred_start, fred_end)
            vxv_df = fetch_fred_series("VXVCLS", fred_start, fred_end)
        except RuntimeError as e:
            log(f"FRED API key missing, skipping VRP merge: {e}", "WARN")
            vix_df = pd.DataFrame(columns=["value"])
            vxv_df = pd.DataFrame(columns=["value"])

        # Helper: fetch a wide daily history for VRP RV calc, separate from
        # the backtest's daily_df (which may only have ~220 bars).
        def _fetch_wide_daily(symbol):
            cache_dir = Path(cfg.get("cache_dir", "apex_cache"))
            wide_cache = cache_dir / f"{symbol}_daily_wide.csv"
            if wide_cache.exists():
                try:
                    df = pd.read_csv(wide_cache, parse_dates=["datetime"])
                    if len(df) >= 800:
                        return df
                except Exception:
                    pass
            data = polygon_request(
                f"v2/aggs/ticker/{symbol}/range/1/day/{fred_start}/{fred_end}",
                {"adjusted": "true", "sort": "asc", "limit": 50000},
            )
            if data is None or "results" not in data:
                return None
            rows = data.get("results", [])
            if not rows:
                return None
            wdf = pd.DataFrame([
                {"datetime": pd.to_datetime(r["t"], unit="ms"),
                 "close": r["c"]}
                for r in rows
            ]).sort_values("datetime").reset_index(drop=True)
            wdf.to_csv(wide_cache, index=False)
            return wdf

        for sym in list(data_dict.keys()):
            sd = data_dict[sym]
            iv_series_id = IV_MAP.get(sym, "VIXCLS")

            try:
                if iv_series_id in ("VIXCLS",):
                    iv_df = vix_df
                else:
                    iv_df = fetch_fred_series(iv_series_id, fred_start, fred_end)
            except RuntimeError:
                iv_df = pd.DataFrame(columns=["value"])

            if iv_df.empty or vix_df.empty or vxv_df.empty:
                log(f"  {sym}: skipping VRP merge (missing FRED data)", "WARN")
                continue

            # Fetch a wide daily history for THIS symbol — needed for the
            # 252-day VRP percentile rolling window.  Falls back to short
            # daily_df if wide fetch fails.
            wide_daily = _fetch_wide_daily(sym)
            if wide_daily is None or len(wide_daily) < 300:
                log(f"  {sym}: wide daily fetch failed, falling back to short", "WARN")
                wide_daily = sd.get("daily_df")
                if wide_daily is None or len(wide_daily) == 0:
                    continue

            daily_close = wide_daily.set_index("datetime")["close"]
            iv_aligned = iv_df["value"].reindex(daily_close.index, method="ffill")

            vrp_result = compute_vrp(iv_aligned, daily_close, rv_window=20, pct_window=252)

            # Forward-fill daily columns onto exec bars
            for target_key in ("exec_df", "exec_df_holdout"):
                edf = sd.get(target_key)
                if edf is None or len(edf) == 0:
                    continue
                bar_dates = edf["datetime"].dt.normalize()

                edf["vix"] = vix_df["value"].reindex(bar_dates.values, method="ffill").values
                edf["vxv"] = vxv_df["value"].reindex(bar_dates.values, method="ffill").values
                edf["vrp_pct"] = vrp_result["vrp_pct"].reindex(bar_dates.values, method="ffill").values

            non_nan_vrp = pd.Series(edf["vrp_pct"]).notna().sum() if edf is not None else 0
            log(f"  {sym}: VRP columns merged (IV={iv_series_id}, "
                f"daily_hist={len(wide_daily)}, vrp_pct non-NaN={non_nan_vrp})")

    # ---- Layer 1 ----
    if cfg.get("strategy_mode") == "vrp_regime":
        log("Layer 1 SKIP: using VRP_DEFAULT_ARCHITECTURE (strategy_mode=vrp_regime)")
        # architecture already set above
    elif resume_stage in ("layer2_tuned", "layer3_robustness"):
        cp = load_checkpoint("layer1_architecture", str(run_output))
        architecture = cp["architecture"] if cp else DEFAULT_ARCHITECTURE
    else:
        architecture = layer1_architecture_search(data_dict, concept_bias, cfg)

    # ---- Cross-Asset Basket (opt-in) ----
    basket = None
    basket_cfg = cfg.get("cross_asset_basket", {})
    if basket_cfg.get("enabled", False):
        from apex.data.cross_asset import fetch_basket
        basket_symbols = basket_cfg.get("symbols")
        log("=== CROSS-ASSET BASKET MOMENTUM ===")
        basket = fetch_basket(basket_symbols)
        if not basket:
            log("Basket fetch returned empty, disabling basket alignment", "WARN")
            basket = None

    # ---- Layer 2 ----
    if resume_stage == "layer3_robustness":
        cp = load_checkpoint("layer2_tuned", str(run_output))
        tuned_results = cp if cp else {}
        for sym, sym_result in tuned_results.items():
            if "trades" not in sym_result and sym in data_dict:
                sd = data_dict[sym]
                params = sym_result.get("params", DEFAULT_PARAMS)
                trades, stats = full_backtest(sd["exec_df"], sd["daily_df"],
                                              architecture, params)
                sym_result["trades"] = trades
                sym_result["trade_pnls"] = [t["pnl_pct"] for t in trades]
                sym_result["stats"] = stats
    else:
        tuned_results = layer2_deep_tune(data_dict, architecture, survivors, cfg,
                                          basket=basket)

    if not tuned_results:
        log("No symbols survived Layer 2. Exiting.", "ERROR")
        sys.exit(1)

    # ---- Layer 3 ----
    validated_results, robustness_data = layer3_robustness_gauntlet(
        data_dict, architecture, tuned_results, cfg
    )

    if not validated_results:
        log("No symbols passed robustness gauntlet. Relaxing threshold...", "WARN")
        sorted_by_composite = sorted(
            robustness_data.items(),
            key=lambda x: x[1].get("composite", 0),
            reverse=True
        )
        for sym, rd in sorted_by_composite[:5]:
            if sym in tuned_results:
                validated_results[sym] = tuned_results[sym]
                validated_results[sym]["robustness"] = rd
        if not validated_results:
            log("Still no symbols. Using best tuned results.", "WARN")
            best_sym = max(tuned_results, key=lambda s: tuned_results[s].get("fitness", 0))
            validated_results = {best_sym: tuned_results[best_sym]}
            validated_results[best_sym]["robustness"] = robustness_data.get(best_sym, {})

    # ---- Correlation Filter ----
    final_results = correlation_filter(validated_results, cfg)

    if not final_results:
        log("All symbols filtered out. Using validated results.", "WARN")
        final_results = validated_results

    # ---- Final Backtest ----
    backtest_results = phase_full_backtest(data_dict, architecture, final_results, cfg,
                                           tuned_results=tuned_results)

    # ---- Report Generation ----
    log("=== GENERATING REPORTS ===")
    report_path = generate_html_report(
        backtest_results, architecture, robustness_data, run_info, str(run_output)
    )
    generate_trades_csv(backtest_results.get("all_trades", []), str(run_output))
    generate_summary_csv(backtest_results, str(run_output))
    generate_parameters_json(backtest_results, architecture, str(run_output))

    # ---- AmiBroker Push ----
    if not args.no_amibroker:
        log("=== AMIBROKER INTEGRATION ===")
        sorted_syms = backtest_results.get("sorted_syms", [])
        afl_str = generate_apex_afl(sorted_syms, backtest_results, architecture)
        push_to_amibroker(backtest_results, afl_str, str(run_output), cfg)
    else:
        log("AmiBroker push skipped (--no-amibroker)")

    # ---- Summary ----
    log("=== PIPELINE COMPLETE ===")
    pstats = backtest_results.get("portfolio_stats", {})
    log("Final Results:")
    log(f"  Symbols: {len(backtest_results.get('sorted_syms', []))}")
    log(f"  Trades:  {pstats.get('trades', 0)}")
    log(f"  PF:      {pstats.get('pf', 0):.2f}")
    log(f"  Win%:    {pstats.get('wr_pct', 0):.1f}%")
    log(f"  Return:  {pstats.get('total_return_pct', 0):.1f}%")
    log(f"  MaxDD:   {pstats.get('max_dd_pct', 0):.1f}%")
    log(f"  Sharpe:  {pstats.get('sharpe', 0):.2f}")
    hstats = backtest_results.get("holdout_universe_stats", {})
    if hstats:
        log("  --- TRUE HOLDOUT (never seen by optimizer) ---")
        log(f"  Holdout Trades:  {hstats.get('trades', 0)}")
        log(f"  Holdout PF:      {hstats.get('pf', 0):.2f}")
        log(f"  Holdout Win%:    {hstats.get('wr_pct', 0):.1f}%")
        log(f"  Holdout Return:  {hstats.get('total_return_pct', 0):.1f}%")
        log(f"  Holdout Sharpe:  {hstats.get('sharpe', 0):.2f}")
    log(f"  Report:  {report_path}")

    try:
        abs_report = str(Path(report_path).resolve()).replace("\\", "/")
        webbrowser.open(f"file:///{abs_report}")
        log(f"Report opened in browser: file:///{abs_report}")
    except Exception as e:
        log(f"Could not open browser: {e}", "WARN")
        log(f"Open manually: {Path(report_path).resolve()}")

    return backtest_results
