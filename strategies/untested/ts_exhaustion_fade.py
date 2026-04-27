"""ts_exhaustion_fade — UNTESTED (no real-data OOS).

Status:        NO-REAL-DATA-OOS
Source:        apex/strategies/ts_exhaustion_fade.py (216 LOC)
Added:         2026-04-26 (commit 223090d, "research: add structural reclaim
               strategy prototypes")

WHY UNTESTED
------------
Has 6 unit tests on synthetic toy bars (test_strategy_ts_exhaustion_fade.py)
that assert signal shape under fixed regime conditions. No CPCV, no walk-
forward, no honest 75/25 holdout, no Sharpe number on real SPY/QQQ data.

REQUIRED PROMOTION GATE
-----------------------
1. Run on real SPY 1H bars 2021-01 -> mid-2025 (tune) and mid-2025 -> 2026-Q1
   (holdout). The tuner must never see the holdout window.
2. Layer B CPCV median Sharpe > 1 on holdout, > 50% positive folds.
3. Holdout total return > 0%, MaxDD > -10%, n_trades >= 30.
4. Beat the 77.7% Aryan Optimized credit-spread baseline DSR.

If all four pass, move this manifest to strategies/tested/ with TUNED_PARAMS
locked from the Optuna best trial.
"""

STATUS = "UNTESTED"
SUBSTATUS = "NO-REAL-DATA-OOS"
SOURCE_MODULE = "apex.strategies.ts_exhaustion_fade"
DEPLOYABLE = False


def make_strategy(*_args, **_kwargs):
    raise RuntimeError(
        "ts_exhaustion_fade has no validated parameters. Run the promotion-"
        "gate validation in this file's docstring first, then move to "
        "strategies/tested/ with locked TUNED_PARAMS."
    )
