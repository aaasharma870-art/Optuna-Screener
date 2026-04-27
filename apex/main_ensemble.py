"""Ensemble pipeline orchestration (Phases 12H + 12I).

Wires together:
  * STRATEGY_REGISTRY (the 6 institutional strategies)
  * EnsembleCombiner (risk-parity + regime overlay)
  * Three-layer validation (per-strategy CPCV / portfolio CPCV / walk-forward weights)
  * Ensemble HTML report

The legacy single-strategy pipeline in apex.main.main() is untouched: this
module is invoked only when --ensemble is passed on the CLI.
"""
import csv
import json
import sys
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from apex.config import CACHE_DIR
from apex.logging_util import log


# --------------------------------------------------------------------------- #
# Data preparation                                                            #
# --------------------------------------------------------------------------- #

def prepare_ensemble_data(data_dict: Dict[str, Dict[str, Any]],
                          cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Augment each symbol's exec_df with all columns required by the ensemble.

    Columns added (best-effort -- a missing column is logged WARN and the
    relevant strategies will gracefully skip those bars):

      * vix, vxv, vrp_pct           -- already merged when strategy_mode == "vrp_regime"
      * call_wall, put_wall, gamma_flip, vol_trigger, abs_gamma_strike
                                    -- via apex.data.dealer_levels.ingest_flux_points
                                       (only if cfg.options_gex.enabled)
      * vix_pct, move_pct, ovx_pct  -- 252-day rolling percentiles of VIX/MOVE/OVX
                                       fetched from FRED.
    """
    cache_dir = Path(cfg.get("cache_dir", "apex_cache"))

    # ---- (0) VIX / VXV / VRP percentile (always-on for ensemble) ------------
    try:
        from apex.data.fred_client import fetch_fred_series, IV_MAP
        from apex.regime.vrp import compute_vrp

        # Determine FRED window from the data
        all_dates_v: List[pd.Timestamp] = []
        for sd in data_dict.values():
            for key in ("exec_df", "exec_df_holdout"):
                edf = sd.get(key)
                if edf is not None and len(edf) > 0 and "datetime" in edf.columns:
                    all_dates_v.extend(pd.to_datetime(edf["datetime"]).tolist())
        if all_dates_v:
            min_v = min(all_dates_v)
            max_v = max(all_dates_v)
            fred_start_v = (min_v - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
            fred_end_v = (max_v + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            try:
                vix_df0 = fetch_fred_series("VIXCLS", fred_start_v, fred_end_v)
            except Exception as e:
                log(f"  VIX FRED fetch (vrp prep) failed: {e}", "WARN")
                vix_df0 = pd.DataFrame(columns=["value"])
            try:
                vxv_df0 = fetch_fred_series("VXVCLS", fred_start_v, fred_end_v)
            except Exception as e:
                log(f"  VXV FRED fetch (vrp prep) failed: {e}", "WARN")
                vxv_df0 = pd.DataFrame(columns=["value"])

            if (not vix_df0.empty) and (not vxv_df0.empty):
                for sym, sd in data_dict.items():
                    iv_id = IV_MAP.get(sym, "VIXCLS")
                    if iv_id == "VIXCLS":
                        iv_df0 = vix_df0
                    else:
                        try:
                            iv_df0 = fetch_fred_series(iv_id, fred_start_v, fred_end_v)
                        except Exception:
                            iv_df0 = vix_df0
                    for key in ("exec_df", "exec_df_holdout"):
                        edf = sd.get(key)
                        if edf is None or len(edf) == 0 or "datetime" not in edf.columns:
                            continue
                        bar_dates = pd.to_datetime(edf["datetime"]).dt.normalize()
                        edf["vix"] = vix_df0["value"].reindex(
                            bar_dates.values, method="ffill").values
                        edf["vxv"] = vxv_df0["value"].reindex(
                            bar_dates.values, method="ffill").values
                        # Compute VRP percentile from this symbol's daily close
                        try:
                            edf_unique_dates = bar_dates.drop_duplicates().sort_values()
                            close_daily = edf.groupby(bar_dates)["close"].last()
                            close_daily.index = pd.DatetimeIndex(close_daily.index)
                            iv_aligned = iv_df0["value"].reindex(
                                close_daily.index, method="ffill")
                            vrp_res = compute_vrp(iv_aligned, close_daily,
                                                   rv_window=20, pct_window=252)
                            edf["vrp_pct"] = vrp_res["vrp_pct"].reindex(
                                bar_dates.values, method="ffill").values
                        except Exception as e:
                            log(f"  {sym}: vrp_pct merge failed: {e}", "WARN")
                            edf["vrp_pct"] = float("nan")
                        sd[key] = edf
                log("VIX/VXV/VRP merged onto exec dataframes")
            else:
                log("VIX/VXV/VRP merge skipped (FRED unavailable)", "WARN")
    except Exception as e:
        log(f"VIX/VXV/VRP merge failed: {e}", "WARN")

    # ---- (1) GEX dealer levels (opt-in) -------------------------------------
    options_gex_enabled = cfg.get("options_gex", {}).get("enabled", False)
    if options_gex_enabled:
        try:
            from apex.data.dealer_levels import ingest_flux_points
            for sym, sd in data_dict.items():
                for key in ("exec_df", "exec_df_holdout"):
                    edf = sd.get(key)
                    if edf is None or len(edf) == 0:
                        continue
                    try:
                        sd[key] = ingest_flux_points(edf, sym, cache_dir)
                    except Exception as e:
                        log(f"  {sym}: GEX merge failed for {key}: {e}", "WARN")
        except Exception as e:
            log(f"GEX dealer-levels merge skipped: {e}", "WARN")
    else:
        log("GEX dealer-levels merge skipped (options_gex.enabled=false)")

    # ---- (2) Cross-asset vol percentiles (VIX/MOVE/OVX) ---------------------
    try:
        from apex.data.cross_asset_vol import (
            fetch_move_index, fetch_ovx, compute_vol_percentiles,
        )
        from apex.data.fred_client import fetch_fred_series

        # Choose a wide range covering all symbols' bars
        all_dates: List[pd.Timestamp] = []
        for sd in data_dict.values():
            for key in ("exec_df", "exec_df_holdout"):
                edf = sd.get(key)
                if edf is not None and len(edf) > 0 and "datetime" in edf.columns:
                    all_dates.extend(pd.to_datetime(edf["datetime"]).tolist())
        if not all_dates:
            log("prepare_ensemble_data: no datetimes found; skipping vol percentiles",
                "WARN")
        else:
            min_dt = min(all_dates)
            max_dt = max(all_dates)
            # FRED needs a wider history for the rolling-252 percentile to be valid
            fred_start = (min_dt - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
            fred_end = (max_dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

            try:
                vix_df = fetch_fred_series("VIXCLS", fred_start, fred_end)
            except Exception as e:
                log(f"  VIX FRED fetch failed: {e}", "WARN")
                vix_df = pd.DataFrame(columns=["value"])
            try:
                move_df = fetch_move_index(fred_start, fred_end, cache_dir)
            except Exception as e:
                log(f"  MOVE FRED fetch failed: {e}", "WARN")
                move_df = pd.DataFrame(columns=["value"])
            try:
                ovx_df = fetch_ovx(fred_start, fred_end, cache_dir)
            except Exception as e:
                log(f"  OVX FRED fetch failed: {e}", "WARN")
                ovx_df = pd.DataFrame(columns=["value"])

            if (not vix_df.empty) and (not move_df.empty) and (not ovx_df.empty):
                vix_s = vix_df["value"]
                move_s = move_df["value"]
                ovx_s = ovx_df["value"]
                # Align all three on a common daily index, ffill
                idx = (
                    vix_s.index.union(move_s.index).union(ovx_s.index).sort_values()
                )
                vix_s = vix_s.reindex(idx).ffill()
                move_s = move_s.reindex(idx).ffill()
                ovx_s = ovx_s.reindex(idx).ffill()
                pct_df = compute_vol_percentiles(vix_s, move_s, ovx_s, window=252)

                for sym, sd in data_dict.items():
                    for key in ("exec_df", "exec_df_holdout"):
                        edf = sd.get(key)
                        if edf is None or len(edf) == 0 or "datetime" not in edf.columns:
                            continue
                        bar_dates = pd.to_datetime(edf["datetime"]).dt.normalize()
                        for col in ("vix_pct", "move_pct", "ovx_pct"):
                            edf[col] = pct_df[col].reindex(
                                bar_dates.values, method="ffill",
                            ).values
                        sd[key] = edf
                log("Cross-asset vol percentiles merged (vix_pct/move_pct/ovx_pct)")
            else:
                log("Cross-asset vol percentiles skipped (FRED unavailable)", "WARN")
    except Exception as e:
        log(f"Cross-asset vol percentile merge failed: {e}", "WARN")

    return data_dict


# --------------------------------------------------------------------------- #
# Strategy <-> backtest engine adapter for Layer A                            #
# --------------------------------------------------------------------------- #

def _strategy_to_returns(strategy, data: Dict[str, Any]) -> pd.Series:
    """Run a strategy on a data dict and return per-bar returns.

    Returns shifted positions x next-bar pct change so look-ahead is prevented.
    """
    sigs = strategy.compute_signals(data)
    pos = strategy.compute_position_size(data, sigs)
    edf = data.get("exec_df_1H", pd.DataFrame())
    if edf.empty or "close" not in edf.columns or len(pos) != len(edf):
        return pd.Series(dtype=float)
    px_ret = edf["close"].pct_change().fillna(0.0).values
    pos_arr = np.asarray(pos.values, dtype=float)
    return pd.Series(np.concatenate(([0.0], pos_arr[:-1] * px_ret[1:])))


def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 252 * 7) -> float:
    """Annualized Sharpe assuming per-bar returns at 1H frequency (~7 bars/day RTH)."""
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma <= 1e-12:
        return 0.0
    return mu / sigma * float(np.sqrt(periods_per_year))


# --------------------------------------------------------------------------- #
# Layer A: per-strategy CPCV                                                  #
# --------------------------------------------------------------------------- #

def run_layer_a_validation(strategies: List[Any],
                           data_dict: Dict[str, Dict[str, Any]],
                           cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """For each (strategy, symbol) run a CPCV at the per-strategy returns level.

    Returns a list of row dicts:
      {strategy_name, symbol, n_folds, median_sharpe, iqr_low, iqr_high,
       pct_positive, layer_a_status}.
    """
    from apex.validation.cpcv import cpcv_split

    val_cfg = cfg.get("validation", {}).get("cpcv", {})
    n_blocks = int(val_cfg.get("n_blocks", 8))
    n_test_blocks = int(val_cfg.get("n_test_blocks", 2))
    purge_bars = int(val_cfg.get("purge_bars", 10))

    rows: List[Dict[str, Any]] = []
    for strat in strategies:
        for sym, sd in data_dict.items():
            edf = sd.get("exec_df")
            if edf is None or len(edf) < 100:
                continue
            # Build a strategy data dict
            sym_data = {"exec_df_1H": edf,
                        "regime_state": pd.Series(["R2"] * len(edf))}

            # Compute full per-bar return series ONCE (signals are deterministic)
            try:
                returns = _strategy_to_returns(strat, sym_data)
            except Exception as e:
                log(f"  Layer A: {strat.name}/{sym} failed compute: {e}", "WARN")
                rows.append({
                    "strategy_name": strat.name, "symbol": sym,
                    "n_folds": 0, "median_sharpe": 0.0,
                    "iqr_low": 0.0, "iqr_high": 0.0, "pct_positive": 0.0,
                    "layer_a_status": "ERROR",
                })
                continue

            if len(returns) < 100:
                continue

            sharpes = []
            for _, test_idx in cpcv_split(len(returns), n_blocks=n_blocks,
                                           n_test_blocks=n_test_blocks,
                                           purge_bars=purge_bars):
                if len(test_idx) < 50:
                    continue
                test_ret = returns.iloc[test_idx]
                sharpes.append(_annualized_sharpe(test_ret))

            if not sharpes:
                rows.append({
                    "strategy_name": strat.name, "symbol": sym,
                    "n_folds": 0, "median_sharpe": 0.0,
                    "iqr_low": 0.0, "iqr_high": 0.0, "pct_positive": 0.0,
                    "layer_a_status": "FAIL",
                })
                continue

            arr = np.array(sharpes, dtype=float)
            median = float(np.median(arr))
            iqr_low = float(np.percentile(arr, 25))
            iqr_high = float(np.percentile(arr, 75))
            pct_pos = float((arr > 0).mean())
            status = "PASS" if median >= 0.0 else "FAIL"
            rows.append({
                "strategy_name": strat.name, "symbol": sym,
                "n_folds": len(sharpes),
                "median_sharpe": median, "iqr_low": iqr_low, "iqr_high": iqr_high,
                "pct_positive": pct_pos,
                "layer_a_status": status,
            })
            log(f"  Layer A: {strat.name}/{sym}: med Sharpe={median:.2f} "
                f"IQR=[{iqr_low:.2f},{iqr_high:.2f}] +%={pct_pos*100:.0f} -> {status}")
    return rows


def write_layer_a_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["strategy_name", "symbol", "n_folds", "median_sharpe",
            "iqr_low", "iqr_high", "pct_positive", "layer_a_status"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


# --------------------------------------------------------------------------- #
# Layer B: portfolio (ensemble) CPCV                                          #
# --------------------------------------------------------------------------- #

def run_layer_b_validation(combiner_result: Dict[str, Any],
                           ref_close: pd.Series,
                           cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Take the combiner's portfolio_position + reference close, derive per-bar
    portfolio returns, and run CPCV at portfolio level."""
    from apex.validation.ensemble_cpcv import evaluate_ensemble_cpcv

    pos = combiner_result.get("portfolio_position")
    if pos is None or len(pos) == 0 or ref_close is None or len(ref_close) == 0:
        return {"n_folds": 0, "error": "no portfolio position"}

    n = min(len(pos), len(ref_close))
    px_ret = pd.Series(ref_close.values[:n]).pct_change().fillna(0.0)
    # Shift positions by 1 bar to enforce: signal at bar i, fill at bar i+1
    pos_shift = pd.Series(pos.values[:n]).shift(1).fillna(0.0)
    portfolio_ret = (pos_shift * px_ret).fillna(0.0)

    val_cfg = cfg.get("validation", {}).get("cpcv", {})
    n_blocks = int(val_cfg.get("n_blocks", 8))
    n_test_blocks = int(val_cfg.get("n_test_blocks", 2))
    purge_bars = int(val_cfg.get("purge_bars", 10))

    res = evaluate_ensemble_cpcv(
        portfolio_ret,
        n_blocks=n_blocks, n_test_blocks=n_test_blocks, purge_bars=purge_bars,
        periods_per_year=252 * 7,  # 1H bars
    )
    # Layer B PASS criterion: median Sharpe > 0.8 AND > 65% folds positive
    median = res.get("sharpe_median", 0.0)
    pct_pos = res.get("sharpe_pct_positive", 0.0)
    res["layer_b_status"] = "PASS" if (median > 0.8 and pct_pos > 0.65) else "FAIL"
    res["portfolio_returns"] = portfolio_ret.tolist()
    return res


def _serialize_layer_b(res: Dict[str, Any]) -> Dict[str, Any]:
    """Make a JSON-safe copy of Layer B results."""
    out = {}
    for k, v in res.items():
        if isinstance(v, (list, str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = _serialize_layer_b(v)
        else:
            try:
                out[k] = float(v)
            except Exception:
                out[k] = str(v)
    return out


# --------------------------------------------------------------------------- #
# Layer C: walk-forward dynamic vs static weights                             #
# --------------------------------------------------------------------------- #

def run_layer_c_validation(per_strategy_returns: Dict[str, pd.Series],
                           ref_dt: pd.Series,
                           cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resample per-strategy per-bar returns to monthly and call
    compare_dynamic_vs_static_weights.
    """
    from apex.validation.walk_forward import compare_dynamic_vs_static_weights

    if not per_strategy_returns or ref_dt is None or len(ref_dt) == 0:
        return {"error": "no inputs", "n_months": 0}

    dt_idx = pd.to_datetime(ref_dt.values)
    monthly: Dict[str, pd.Series] = {}
    for name, s in per_strategy_returns.items():
        if s is None or len(s) == 0:
            continue
        n = min(len(s), len(dt_idx))
        ser = pd.Series(s.values[:n], index=dt_idx[:n])
        # Compound returns to monthly
        monthly_ret = (1 + ser).resample("ME").prod() - 1
        if len(monthly_ret) < 4:
            continue
        monthly[name] = monthly_ret

    if not monthly:
        return {"error": "no monthly returns", "n_months": 0}

    res = compare_dynamic_vs_static_weights(monthly, warmup_months=6)
    uplift = res.get("uplift")
    if uplift is None:
        res["layer_c_status"] = "UNKNOWN"
    else:
        res["layer_c_status"] = "PASS" if uplift >= 0.05 else "FAIL"
    return res


# --------------------------------------------------------------------------- #
# Combiner runner                                                             #
# --------------------------------------------------------------------------- #

def _resolve_strategy_classes(cfg: Dict[str, Any]) -> List[Any]:
    """Resolve the configured ensemble strategy CLASSES (not instances).

    Triggers registration of all 6 strategies via decorator-on-import, then
    looks up each name configured under ensemble.strategies in the registry.
    """
    # Trigger registration of all 6 strategies (decorators register on import)
    import apex.strategies.vrp_gex_fade  # noqa: F401
    import apex.strategies.opex_gravity  # noqa: F401
    import apex.strategies.vix_term_structure  # noqa: F401
    import apex.strategies.ts_exhaustion_fade  # noqa: F401
    import apex.strategies.vix_liquidity_reclaim  # noqa: F401
    import apex.strategies.institutional_arbitrage_engine_v2  # noqa: F401
    import apex.strategies.advanced_compounder_v11  # noqa: F401
    import apex.strategies.vol_skew_arb  # noqa: F401
    import apex.strategies.smc_structural  # noqa: F401
    import apex.strategies.cross_asset_vol_overlay  # noqa: F401
    from apex.strategies import STRATEGY_REGISTRY

    ens_cfg = cfg.get("ensemble", {})
    names = ens_cfg.get("strategies") or list(STRATEGY_REGISTRY.keys())

    classes = []
    for name in names:
        cls = STRATEGY_REGISTRY.get(name)
        if cls is None:
            log(f"  Ensemble: strategy '{name}' not in registry, skipping", "WARN")
            continue
        classes.append(cls)
        log(f"  Ensemble: resolved strategy class '{name}'")
    return classes


def _build_strategies(cfg: Dict[str, Any]) -> List[Any]:
    """Instantiate the curated ensemble strategies from the registry (defaults)."""
    classes = _resolve_strategy_classes(cfg)
    strategies = []
    for cls in classes:
        try:
            strategies.append(cls())
        except Exception as e:
            log(f"  Ensemble: failed to instantiate '{cls.name}': {e}", "WARN")
    return strategies


# --------------------------------------------------------------------------- #
# Phase 13: per-strategy Optuna tuning                                        #
# --------------------------------------------------------------------------- #

# Maps the CLI / config budget label to a per-strategy trial count.
ENSEMBLE_TUNING_TRIALS_BY_BUDGET: Dict[str, int] = {
    "light":  25,
    "medium": 50,
    "heavy":  150,
}


def _budget_label(cfg: Dict[str, Any]) -> str:
    """Infer the budget label from the active config's optimization.arch_trials.

    The CLI rewrites optimization.arch_trials from budget_profiles.<budget>
    before run_ensemble_pipeline is called, so we map back via that count.
    """
    bp = cfg.get("budget_profiles", {})
    arch = int(cfg.get("optimization", {}).get("arch_trials", 0))
    for label in ("light", "medium", "heavy"):
        if int(bp.get(label, {}).get("arch_trials", -1)) == arch:
            return label
    return "medium"


def _tuning_n_trials(cfg: Dict[str, Any]) -> int:
    """Resolve the number of Optuna trials per strategy from the budget."""
    label = _budget_label(cfg)
    return ENSEMBLE_TUNING_TRIALS_BY_BUDGET.get(label, 50)


def _tuning_data_per_symbol(data_dict: Dict[str, Dict[str, Any]]
                              ) -> Dict[str, Dict[str, Any]]:
    """Build the {symbol: data dict} map that tune_ensemble_strategies expects.

    Each value is a strategy-shaped dict (exec_df_1H + regime_state).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for sym, sd in data_dict.items():
        edf = sd.get("exec_df")
        if edf is None or len(edf) == 0:
            continue
        out[sym] = {
            "exec_df_1H": edf,
            "regime_state": pd.Series(["R2"] * len(edf)),
            "symbol": sym,
        }
    return out


def _serialize_tuning_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """JSON-safe copy of the tuning results dict."""
    out: Dict[str, Any] = {}
    for name, r in results.items():
        out[name] = {}
        for k, v in r.items():
            if isinstance(v, dict):
                out[name][k] = {
                    kk: (float(vv) if isinstance(vv, (np.floating,)) else vv)
                    for kk, vv in v.items()
                }
            elif isinstance(v, (np.floating,)):
                out[name][k] = float(v)
            elif isinstance(v, (np.integer,)):
                out[name][k] = int(v)
            else:
                out[name][k] = v
    return out


def _pick_primary_symbol(data_dict: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Pick SPY if available (the curated ensemble is SPY-centric); else first symbol."""
    if "SPY" in data_dict:
        return "SPY"
    for s in data_dict:
        return s
    return None


def _build_strategy_data(sym_data: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a single symbol's tune-window exec_df into the dict the combiner expects."""
    edf = sym_data.get("exec_df")
    n = len(edf) if edf is not None else 0
    return {
        "exec_df_1H": edf if edf is not None else pd.DataFrame(),
        "regime_state": pd.Series(["R2"] * n) if n > 0 else pd.Series(dtype=object),
    }


# --------------------------------------------------------------------------- #
# Main ensemble pipeline                                                      #
# --------------------------------------------------------------------------- #

def run_ensemble_pipeline(data_dict: Dict[str, Dict[str, Any]],
                          cfg: Dict[str, Any], run_info: Dict[str, Any],
                          run_output: Path,
                          no_amibroker: bool = True) -> Dict[str, Any]:
    """End-to-end ensemble run.

    1. Augment data_dict with required columns
    2. Build strategies from STRATEGY_REGISTRY
    3. Layer A: per-strategy CPCV
    4. Build EnsembleCombiner, run on tune window of primary symbol
    5. Layer B: portfolio CPCV
    6. Layer C: walk-forward dynamic vs static weights
    7. Generate ensemble HTML report + JSON/CSV outputs
    """
    log("=" * 60)
    log("ENSEMBLE PIPELINE (Phase 12H/12I)")
    log("=" * 60)

    # ---- Phase 12H/2: data augmentation ----
    log("=== ENSEMBLE: prepare_ensemble_data ===")
    data_dict = prepare_ensemble_data(data_dict, cfg)

    # ---- Phase 12H/1: resolve strategy classes ----
    log("=== ENSEMBLE: resolving strategy classes ===")
    strategy_classes = _resolve_strategy_classes(cfg)
    if not strategy_classes:
        log("No strategies available -- aborting", "ERROR")
        return {}

    # ---- Phase 13: per-strategy Optuna tuning (CPCV-scored) ----
    ens_cfg = cfg.get("ensemble", {})
    tune_strategies_flag = bool(ens_cfg.get("tune_strategies", True))
    tuning_results: Dict[str, Dict[str, Any]] = {}

    if tune_strategies_flag:
        from apex.optimize.ensemble_tuning import tune_ensemble_strategies

        n_trials = _tuning_n_trials(cfg)
        budget_label = _budget_label(cfg)
        log(f"=== ENSEMBLE: Phase 13 per-strategy Optuna tuning "
            f"(budget={budget_label}, {n_trials} trials/strategy) ===")
        tuning_data = _tuning_data_per_symbol(data_dict)
        try:
            tuning_results = tune_ensemble_strategies(
                strategy_classes, tuning_data,
                n_trials_per_strategy=n_trials,
            )
        except Exception as e:
            log(f"Phase 13 tuning failed: {e}", "ERROR")
            traceback.print_exc()
            tuning_results = {}

        # Persist tuning results
        try:
            (run_output / "ensemble_tuning_results.json").write_text(
                json.dumps(_serialize_tuning_results(tuning_results), indent=2)
            )
            log(f"  Tuning results: {run_output / 'ensemble_tuning_results.json'}")
        except Exception as e:
            log(f"  Could not write tuning results: {e}", "WARN")

        # Tuning summary printout
        log("--- Phase 13 tuning summary ---")
        for cls in strategy_classes:
            tuned = tuning_results.get(cls.name, {})
            bp = tuned.get("best_params", {})
            bs = tuned.get("best_sharpe", float("nan"))
            if bp:
                log(f"  {cls.name:30s} best CPCV Sharpe={bs:.3f}")
                for k, v in bp.items():
                    if isinstance(v, float):
                        log(f"      {k:30s} = {v:.4f}")
                    else:
                        log(f"      {k:30s} = {v}")
            else:
                log(f"  {cls.name:30s} (no tunable params or tuning skipped)")
    else:
        log("=== ENSEMBLE: Phase 13 tuning DISABLED "
            "(ensemble.tune_strategies=false) -- using strategy defaults ===")

    # Instantiate strategies, injecting tuned params when available
    strategies: List[Any] = []
    for cls in strategy_classes:
        tuned = tuning_results.get(cls.name, {})
        best_params = tuned.get("best_params", {}) or {}
        try:
            instance = cls(params=best_params) if best_params else cls()
            strategies.append(instance)
            log(f"  Ensemble: loaded strategy '{cls.name}'"
                f"{' (tuned)' if best_params else ' (defaults)'}")
        except Exception as e:
            log(f"  Ensemble: failed to instantiate '{cls.name}': {e}", "WARN")
    if not strategies:
        log("No strategies could be instantiated -- aborting", "ERROR")
        return {}

    # ---- Layer A: per-strategy CPCV ----
    log("=== ENSEMBLE: Layer A (per-strategy CPCV) ===")
    layer_a_rows: List[Dict[str, Any]] = []
    try:
        layer_a_rows = run_layer_a_validation(strategies, data_dict, cfg)
    except Exception as e:
        log(f"Layer A failed: {e}", "ERROR")
        traceback.print_exc()
    layer_a_csv = run_output / "strategy_layer_a_results.csv"
    write_layer_a_csv(layer_a_rows, layer_a_csv)
    log(f"  Layer A CSV: {layer_a_csv}")

    # ---- Build combiner & run ----
    primary = _pick_primary_symbol(data_dict)
    if primary is None:
        log("No primary symbol available for ensemble -- aborting", "ERROR")
        return {}
    log(f"=== ENSEMBLE: combiner on primary symbol = {primary} ===")
    primary_data = _build_strategy_data(data_dict[primary])
    if primary_data["exec_df_1H"].empty:
        log("Primary symbol has empty exec_df -- aborting", "ERROR")
        return {}

    from apex.ensemble.combiner import EnsembleCombiner
    ens_cfg = cfg.get("ensemble", {})
    combiner = EnsembleCombiner(
        strategies,
        max_weight=float(ens_cfg.get("max_weight", 0.30)),
        vol_lookback_days=int(ens_cfg.get("vol_lookback_days", 60)),
        size_change_threshold=float(ens_cfg.get("size_change_threshold", 0.10)),
    )

    try:
        combiner_result = combiner.run(primary_data)
    except Exception as e:
        log(f"Combiner crashed on {primary}: {e}", "ERROR")
        traceback.print_exc()
        return {}

    weights = combiner_result.get("weights", {})
    log("  Final weights:")
    for name, w in weights.items():
        log(f"    {name:30s} -> {w:.3f}")

    ref_close = primary_data["exec_df_1H"]["close"]
    ref_dt = primary_data["exec_df_1H"]["datetime"]

    # ---- Layer B (TUNE window): portfolio CPCV ----
    log("=== ENSEMBLE: Layer B (TUNE window CPCV) ===")
    layer_b: Dict[str, Any] = {}
    try:
        layer_b = run_layer_b_validation(combiner_result, ref_close, cfg)
    except Exception as e:
        log(f"Layer B failed: {e}", "ERROR")
        traceback.print_exc()
        layer_b = {"error": str(e)}
    layer_b_serial = _serialize_layer_b(layer_b)
    (run_output / "ensemble_layer_b_results.json").write_text(
        json.dumps(layer_b_serial, indent=2)
    )
    median_b = layer_b.get("sharpe_median", 0.0)
    iqr_b = layer_b.get("sharpe_iqr", (0.0, 0.0))
    pct_b = layer_b.get("sharpe_pct_positive", 0.0)
    log(f"  Layer B (tune):    median Sharpe={median_b:.2f} "
        f"IQR=[{iqr_b[0]:.2f},{iqr_b[1]:.2f}] +%={pct_b*100:.0f} "
        f"-> {layer_b.get('layer_b_status','?')}")

    # ---- Layer B (HOLDOUT window): TRUE OOS portfolio CPCV ----
    log("=== ENSEMBLE: Layer B (HOLDOUT window CPCV) "
        "-- TRUE OOS, never seen by tuner ===")
    layer_b_holdout: Dict[str, Any] = {}
    holdout_combiner_result: Dict[str, Any] = {}
    holdout_ref_close: Optional[pd.Series] = None
    holdout_ref_dt: Optional[pd.Series] = None
    holdout_data_per_symbol: Dict[str, Dict[str, Any]] = {}
    primary_holdout_df = data_dict.get(primary, {}).get("exec_df_holdout")
    if primary_holdout_df is None or len(primary_holdout_df) < 100:
        log(f"  Holdout window too short for {primary} "
            f"(need >=100 bars, got {0 if primary_holdout_df is None else len(primary_holdout_df)}) "
            f"-- skipping Layer B holdout", "WARN")
        layer_b_holdout = {"error": "holdout too short", "skipped": True}
    else:
        # Build holdout-shaped data dict for the combiner
        for sym, sd in data_dict.items():
            hdf = sd.get("exec_df_holdout")
            if hdf is None or len(hdf) <= 100:
                continue
            n_h = len(hdf)
            holdout_data_per_symbol[sym] = {
                "exec_df_1H": hdf,
                "regime_state": sd.get(
                    "regime_state_holdout",
                    pd.Series(["UNKNOWN"] * n_h),
                ),
                "symbol": sym,
            }
        # Run the combiner on the holdout window of the primary symbol
        if primary not in holdout_data_per_symbol:
            log(f"  Primary {primary} dropped from holdout map (too short) "
                f"-- skipping Layer B holdout", "WARN")
            layer_b_holdout = {"error": "primary holdout missing", "skipped": True}
        else:
            try:
                holdout_combiner_result = combiner.run(
                    holdout_data_per_symbol[primary]
                )
                holdout_ref_close = holdout_data_per_symbol[primary]["exec_df_1H"]["close"]
                holdout_ref_dt = holdout_data_per_symbol[primary]["exec_df_1H"]["datetime"]
                layer_b_holdout = run_layer_b_validation(
                    holdout_combiner_result, holdout_ref_close, cfg,
                )
            except Exception as e:
                log(f"  Layer B (holdout) failed: {e}", "ERROR")
                traceback.print_exc()
                layer_b_holdout = {"error": str(e)}
    layer_b_holdout_serial = _serialize_layer_b(layer_b_holdout)
    (run_output / "ensemble_layer_b_holdout_results.json").write_text(
        json.dumps(layer_b_holdout_serial, indent=2)
    )
    median_b_h = layer_b_holdout.get("sharpe_median", 0.0) or 0.0
    iqr_b_h = layer_b_holdout.get("sharpe_iqr", (0.0, 0.0)) or (0.0, 0.0)
    pct_b_h = layer_b_holdout.get("sharpe_pct_positive", 0.0) or 0.0
    if layer_b_holdout.get("skipped"):
        log("  Layer B (holdout): SKIPPED (insufficient holdout data)")
        sharpe_decay = float("nan")
    else:
        log(f"  Layer B (holdout): median Sharpe={median_b_h:.2f} "
            f"IQR=[{iqr_b_h[0]:.2f},{iqr_b_h[1]:.2f}] +%={pct_b_h*100:.0f} "
            f"-> {layer_b_holdout.get('layer_b_status','?')}")
        sharpe_decay = float(median_b_h) - float(median_b)
        log(f"  Holdout vs Tune Sharpe gap: {sharpe_decay:+.2f} "
            f"({'decay' if sharpe_decay < 0 else 'lift'})")

    # ---- Layer C: walk-forward weights ----
    log("=== ENSEMBLE: Layer C (walk-forward weights) ===")
    per_strategy_returns_bar: Dict[str, pd.Series] = {}
    px_ret = pd.Series(ref_close.values).pct_change().fillna(0.0)
    for name, pos in combiner_result.get("per_strategy_positions", {}).items():
        if name == "cross_asset_vol_overlay":
            continue
        n = min(len(pos), len(px_ret))
        pos_shift = pd.Series(pos.values[:n]).shift(1).fillna(0.0)
        per_strategy_returns_bar[name] = (pos_shift * px_ret.values[:n]).fillna(0.0)

    layer_c: Dict[str, Any] = {}
    try:
        layer_c = run_layer_c_validation(per_strategy_returns_bar, ref_dt, cfg)
    except Exception as e:
        log(f"Layer C failed: {e}", "ERROR")
        traceback.print_exc()
        layer_c = {"error": str(e)}
    (run_output / "ensemble_layer_c_results.json").write_text(
        json.dumps(_serialize_layer_b(layer_c), indent=2)
    )
    if "uplift" in layer_c:
        log(f"  Layer C: dyn={layer_c.get('dynamic_sharpe',0.0):.2f} "
            f"static={layer_c.get('static_sharpe',0.0):.2f} "
            f"uplift={layer_c.get('uplift',0.0):.2f} -> {layer_c.get('layer_c_status','?')}")
    else:
        log(f"  Layer C error: {layer_c.get('error','?')}")

    # ---- PnL stats (TUNE + HOLDOUT) ----
    from apex.ensemble.pnl import compute_pnl_stats

    pnl_periods_per_year = 252 * 7  # ~7 RTH bars/day at 1H

    def _pnl_table(label: str,
                   ref_close_local: pd.Series,
                   per_strategy_pos: Dict[str, pd.Series],
                   portfolio_pos: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Compute and pretty-print PnL per strategy + ensemble."""
        out: Dict[str, Dict[str, Any]] = {}
        log(f"=== ENSEMBLE: PnL ({label}) ===")
        log(f"  {'Strategy':30s} {'Trades':>7s} {'WinRate%':>9s} "
            f"{'TotalRet%':>10s} {'MaxDD%':>9s} {'Sharpe':>8s} {'Calmar':>8s}")
        for name, pos in per_strategy_pos.items():
            if name == "cross_asset_vol_overlay":
                continue
            stats = compute_pnl_stats(
                pd.Series(pos), pd.Series(ref_close_local),
                periods_per_year=pnl_periods_per_year,
            )
            out[name] = stats
            if "error" in stats:
                log(f"  {name:30s}  (error: {stats['error']})")
                continue
            log(f"  {name:30s} {stats['n_trades']:>7d} "
                f"{stats['win_rate_pct']:>9.1f} "
                f"{stats['total_return_pct']:>+10.2f} "
                f"{stats['max_dd_pct']:>+9.2f} "
                f"{stats['sharpe_annualized']:>8.2f} "
                f"{stats['calmar']:>8.2f}")
        port_stats = compute_pnl_stats(
            pd.Series(portfolio_pos), pd.Series(ref_close_local),
            periods_per_year=pnl_periods_per_year,
        )
        out["__portfolio__"] = port_stats
        if "error" not in port_stats:
            log(f"  {'PORTFOLIO (combined)':30s} {port_stats['n_trades']:>7d} "
                f"{port_stats['win_rate_pct']:>9.1f} "
                f"{port_stats['total_return_pct']:>+10.2f} "
                f"{port_stats['max_dd_pct']:>+9.2f} "
                f"{port_stats['sharpe_annualized']:>8.2f} "
                f"{port_stats['calmar']:>8.2f}")
        return out

    pnl_tune = _pnl_table(
        "TUNE window",
        ref_close,
        combiner_result.get("per_strategy_positions", {}),
        combiner_result.get("portfolio_position", pd.Series(dtype=float)),
    )

    if holdout_combiner_result and holdout_ref_close is not None:
        pnl_holdout = _pnl_table(
            "HOLDOUT window -- TRUE OOS",
            holdout_ref_close,
            holdout_combiner_result.get("per_strategy_positions", {}),
            holdout_combiner_result.get(
                "portfolio_position", pd.Series(dtype=float)
            ),
        )
        # Decay diagnostic
        port_t = pnl_tune.get("__portfolio__", {}) or {}
        port_h = pnl_holdout.get("__portfolio__", {}) or {}
        if "error" not in port_t and "error" not in port_h:
            sharpe_drop = float(port_h.get("sharpe_annualized", 0.0)) - \
                          float(port_t.get("sharpe_annualized", 0.0))
            return_drop = float(port_h.get("total_return_pct", 0.0)) - \
                          float(port_t.get("total_return_pct", 0.0))
            log(f"  Holdout-vs-Tune decay: Sharpe {sharpe_drop:+.2f}, "
                f"Return {return_drop:+.2f}%")
    else:
        pnl_holdout = {}
        log("=== ENSEMBLE: PnL (HOLDOUT) skipped (no holdout combiner result) ===")

    # Persist PnL data (full equity curves included)
    try:
        pnl_payload = {
            "tune": {k: _serialize_layer_b(v) for k, v in pnl_tune.items()},
            "holdout": {k: _serialize_layer_b(v) for k, v in pnl_holdout.items()},
            "tune_split_index": len(ref_close) if ref_close is not None else 0,
        }
        (run_output / "ensemble_pnl.json").write_text(
            json.dumps(pnl_payload, indent=2)
        )
        log(f"  PnL JSON: {run_output / 'ensemble_pnl.json'}")
    except Exception as e:
        log(f"  Could not write PnL JSON: {e}", "WARN")

    # ---- Headline summary ----
    log("=== ENSEMBLE SUMMARY ===")
    log("Per-strategy weights + Layer A status:")
    layer_a_by_strat: Dict[str, str] = {}
    for r in layer_a_rows:
        prev = layer_a_by_strat.get(r["strategy_name"], "PASS")
        if r["layer_a_status"] != "PASS":
            layer_a_by_strat[r["strategy_name"]] = r["layer_a_status"]
        else:
            layer_a_by_strat.setdefault(r["strategy_name"], "PASS")
    for s in strategies:
        w = weights.get(s.name, 0.0)
        st = layer_a_by_strat.get(s.name, "N/A")
        log(f"  {s.name:30s} w={w:.3f}  Layer A: {st}")
    log(f"Layer B (tune)    median Sharpe = {median_b:.2f}, IQR = "
        f"[{iqr_b[0]:.2f}, {iqr_b[1]:.2f}], +%folds = {pct_b*100:.0f}")
    if not layer_b_holdout.get("skipped") and "error" not in layer_b_holdout:
        log(f"Layer B (holdout) median Sharpe = {median_b_h:.2f}, IQR = "
            f"[{iqr_b_h[0]:.2f}, {iqr_b_h[1]:.2f}], +%folds = {pct_b_h*100:.0f}")
    if "uplift" in layer_c:
        log(f"Layer C dynamic={layer_c['dynamic_sharpe']:.2f} "
            f"static={layer_c['static_sharpe']:.2f} "
            f"uplift={layer_c['uplift']:+.2f}")

    # ---- HTML report ----
    log("=== ENSEMBLE: generating HTML report ===")
    results_bundle = {
        "primary_symbol": primary,
        "weights": dict(weights),
        "strategies": [s.name for s in strategies],
        "layer_a_rows": layer_a_rows,
        "layer_a_by_strategy": layer_a_by_strat,
        "layer_b": layer_b_serial,
        "layer_b_holdout": layer_b_holdout_serial,
        "layer_c": _serialize_layer_b(layer_c),
        "combiner_trades": combiner_result.get("trades", []),
        "current_regime": combiner_result.get("current_regime", "UNKNOWN"),
        "portfolio_position": list(combiner_result.get("portfolio_position", [])),
        "ref_dt": [str(d) for d in ref_dt.tolist()],
        "ref_close": [float(c) for c in ref_close.tolist()],
        "per_strategy_positions": {
            n: list(p) for n, p in combiner_result.get("per_strategy_positions", {}).items()
        },
        "pnl_tune": {k: _serialize_layer_b(v) for k, v in pnl_tune.items()},
        "pnl_holdout": {k: _serialize_layer_b(v) for k, v in pnl_holdout.items()},
        "run_info": run_info,
    }

    try:
        from apex.report.ensemble_report import generate_ensemble_report
        report_path = generate_ensemble_report(results_bundle, str(run_output))
        log(f"  Report: {report_path}")
        try:
            abs_report = str(Path(report_path).resolve()).replace("\\", "/")
            webbrowser.open(f"file:///{abs_report}")
            log(f"  Opened: file:///{abs_report}")
        except Exception as e:
            log(f"Could not open browser: {e}", "WARN")
    except Exception as e:
        log(f"HTML report generation failed: {e}", "ERROR")
        traceback.print_exc()

    if not no_amibroker:
        log("AmiBroker push not implemented for ensemble mode; skipping", "WARN")

    log("=== ENSEMBLE PIPELINE COMPLETE ===")
    return results_bundle
