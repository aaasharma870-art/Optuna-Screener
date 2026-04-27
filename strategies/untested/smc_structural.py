"""smc_structural — FAILED on real-data holdout.

Status:        FAILED
Source:        apex/strategies/smc_structural.py
Validated at:  Phase 15 honest holdout (2026-04-26)

REAL-DATA RESULTS
-----------------
Tune window total return:    -2.35%
Holdout window total return: -8.49%

Lost money in BOTH windows. Correctly excluded from the production preset
(strategies/production/vix_term_structure_v1.py mentions this exclusion in
its known-risks section).

WHY IT FAILED
-------------
Smart Money Concepts (order blocks, liquidity sweeps, fair-value gaps) are
visually compelling on charts but the rules are notoriously discretionary
when coded. The systematic version produced low-quality entries with weak
mean-reversion follow-through on SPY/QQQ 1H bars in 2021-2026.

DO NOT PROMOTE
--------------
This is not an "untuned" failure — it ran the full Phase 13 Optuna search
and still lost money OOS. Re-tuning on the same window space is a textbook
overfit risk. If you want SMC exposure, build it as a confirmation filter
on top of the term-structure strategy, not as a standalone signal source.
"""

STATUS = "FAILED"
SUBSTATUS = "LOST-MONEY-OOS"
SOURCE_MODULE = "apex.strategies.smc_structural"
DEPLOYABLE = False
TUNE_RETURN_PCT = -2.35
HOLDOUT_RETURN_PCT = -8.49


def make_strategy(*_args, **_kwargs):
    raise RuntimeError(
        "smc_structural FAILED Phase-15 holdout (-8.49% on unseen data). "
        "Do not deploy. See manifest docstring."
    )
