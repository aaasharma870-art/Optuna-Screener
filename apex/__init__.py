"""
Optuna Screener Pipeline
========================
A generic, Optuna-driven research pipeline for systematic trading strategies.

This package re-exports all public names so that ``import apex`` followed by
``apex.full_backtest(...)`` etc. continues to work as before.
"""

# ---- Configuration ----
from apex.config import load_config, CFG, POLYGON_KEY, CACHE_DIR, OUTPUT_DIR
from apex.config import POLYGON_SLEEP, MAX_RETRIES, RETRY_WAIT, POLYGON_BASE
from apex.config import FORCED_SYMBOLS

# ---- Logging utilities ----
from apex.logging_util import log, eta_str

# ---- Polygon REST client ----
from apex.data.polygon_client import polygon_request, fetch_daily, fetch_bars

# ---- Technical indicators ----
from apex.indicators.basics import (
    compute_ema, compute_atr, compute_vwap, compute_rsi, compute_macd,
    compute_bollinger, compute_stochastic, compute_obv, compute_adx,
    compute_cci, compute_williams_r, compute_keltner, compute_volume_surge,
    parkinson_iv_proxy,
)

# ---- Concept parser / registry ----
from apex.util.concept_parser import INDICATOR_REGISTRY, parse_concept

# ---- Sector map ----
from apex.util.sector_map import SECTOR_MAP

# ---- Backtest engine ----
from apex.engine.backtest import (
    compute_indicator_signals, compute_regime, compute_entry_score,
    run_backtest, compute_stats, full_backtest,
    DEFAULT_ARCHITECTURE, DEFAULT_PARAMS,
)

# ---- Checkpoints ----
from apex.util.checkpoints import save_checkpoint, load_checkpoint

# ---- Optimization layers ----
from apex.optimize.layer1 import (
    _compute_fitness, _mini_monte_carlo, _select_indicators_biased,
    architecture_trial, layer1_architecture_search,
)
from apex.optimize.layer2 import deep_tune_objective, layer2_deep_tune
from apex.optimize.layer3 import (
    monte_carlo_validate, noise_injection_test, regime_stress_test,
    param_sensitivity_test, layer3_robustness_gauntlet,
)

# ---- Portfolio / correlation ----
from apex.engine.portfolio import correlation_filter, phase_full_backtest

# ---- Reports ----
from apex.report.html_report import generate_html_report
from apex.report.csv_json import (
    generate_trades_csv, generate_summary_csv, generate_parameters_json,
)
from apex.report.amibroker import generate_apex_afl, push_to_amibroker

# ---- Main pipeline ----
from apex.main import phase1_universe, phase2_quick_screen, phase3_fetch_data, main

# ---- Suppress warnings (as original) ----
import warnings
warnings.filterwarnings("ignore")

# ---- Optuna setup (as original) ----
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None
