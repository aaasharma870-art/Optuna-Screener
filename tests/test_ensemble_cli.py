"""Tests for the --ensemble CLI flag and dispatch.

Verifies:
  * argparse accepts --ensemble (smoke check)
  * config schema includes the ensemble block (Task 75)
  * the ensemble dispatch is wired through apex.main_ensemble
  * legacy non-ensemble path is unaffected
"""
import argparse
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


def test_apex_config_has_ensemble_block():
    cfg = json.loads((REPO / "apex_config.json").read_text())
    ens = cfg.get("ensemble")
    assert ens is not None, "apex_config.json must define an 'ensemble' block"
    assert ens.get("enabled") in (False, True)
    assert isinstance(ens.get("strategies"), list)
    # Default ensemble must match the production-validated preset.
    # Untested strategies live in strategies/untested/ and must NOT be in
    # the default list — gating them prevents the ~200hr options-chain
    # ingest and re-introducing smc_structural which lost money OOS.
    validated = {"vix_term_structure", "cross_asset_vol_overlay"}
    untested = {"vrp_gex_fade", "opex_gravity", "vol_skew_arb",
                "smc_structural", "ts_exhaustion_fade",
                "vix_liquidity_reclaim", "institutional_arbitrage_engine_v2",
                "advanced_compounder_v11"}
    assert set(ens["strategies"]) == validated
    assert untested.isdisjoint(set(ens["strategies"]))
    assert ens.get("max_weight") == 1.00
    assert ens.get("vol_lookback_days") == 60
    assert ens.get("size_change_threshold") == 0.10


def test_main_ensemble_module_importable():
    """Ensemble pipeline module must be importable for the CLI dispatch."""
    from apex import main_ensemble
    assert hasattr(main_ensemble, "run_ensemble_pipeline")
    assert hasattr(main_ensemble, "prepare_ensemble_data")
    assert hasattr(main_ensemble, "run_layer_a_validation")
    assert hasattr(main_ensemble, "run_layer_b_validation")
    assert hasattr(main_ensemble, "run_layer_c_validation")


def test_main_argparse_accepts_ensemble_flag():
    """Quickly verify the --ensemble flag is present in apex.main argparse."""
    src = (REPO / "apex" / "main.py").read_text()
    assert "--ensemble" in src
    assert "args.ensemble" in src


def test_main_argparse_accepts_strategy_screener_flags():
    """Quickly verify the strategy screener CLI is wired."""
    src = (REPO / "apex" / "main.py").read_text()
    assert "--screen-strategy" in src
    assert "--screen-sp500" in src
    assert "run_strategy_universe_screener" in src


def test_legacy_path_still_intact():
    """The legacy DEFAULT_ARCHITECTURE / DEFAULT_PARAMS must NOT be touched."""
    from apex.engine.backtest import DEFAULT_ARCHITECTURE, DEFAULT_PARAMS
    # Defaults exist and are dicts
    assert isinstance(DEFAULT_ARCHITECTURE, dict)
    assert isinstance(DEFAULT_PARAMS, dict)
    # Spot-check known canonical defaults: architecture has indicators + exit methods
    assert "indicators" in DEFAULT_ARCHITECTURE
    assert "exit_methods" in DEFAULT_ARCHITECTURE
    # Params include canonical sizing/risk knobs
    assert "atr_period" in DEFAULT_PARAMS or "rsi_period" in DEFAULT_PARAMS
