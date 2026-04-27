"""vix_liquidity_reclaim — UNTESTED (no real-data OOS).

Status:        NO-REAL-DATA-OOS
Source:        apex/strategies/vix_liquidity_reclaim.py (301 LOC)
Added:         2026-04-26 (commit 223090d)

WHY UNTESTED
------------
6 unit tests on synthetic toy bars verify shape (sweep + reclaim + VIX
rollover gate). No real-data CPCV, no walk-forward, no holdout Sharpe.

DESIGN RISK TO RESOLVE BEFORE PROMOTION
---------------------------------------
The strategy mixes two reference levels (prior-day H/L and opening range)
into the same long/short reclaim logic. With both enabled, signal density
will likely be 2-3x higher than either alone. Tuning may collapse to one
reference; the validation should expose that ablation explicitly.

REQUIRED PROMOTION GATE
-----------------------
Same as ts_exhaustion_fade — Phase-15 honest holdout, Sharpe > 1 OOS, MaxDD
> -10%, n_trades >= 30, beat Aryan Optimized DSR baseline.
"""

STATUS = "UNTESTED"
SUBSTATUS = "NO-REAL-DATA-OOS"
SOURCE_MODULE = "apex.strategies.vix_liquidity_reclaim"
DEPLOYABLE = False


def make_strategy(*_args, **_kwargs):
    raise RuntimeError(
        "vix_liquidity_reclaim has no validated parameters. See manifest "
        "docstring for the promotion gate."
    )
