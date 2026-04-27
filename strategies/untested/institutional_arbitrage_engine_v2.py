"""institutional_arbitrage_engine_v2 — UNTESTED (no real-data OOS).

Status:        NO-REAL-DATA-OOS
Source:        apex/strategies/institutional_arbitrage_engine_v2.py (270 LOC)
Added:         2026-04-26 (commit 223090d)

WHY UNTESTED
------------
6 unit tests on synthetic toy bars verify each engine in isolation. No
real-data CPCV, no holdout, no Sharpe number. The combined 3-engine
behaviour on real bars has never been measured.

DESIGN RISK TO RESOLVE BEFORE PROMOTION
---------------------------------------
Three engines (calm-contango VWAP fade, regime-adaptive momentum, OPEX pin
gravity) can issue conflicting signals on the same bar. Current code lets
same-bar entry override a same-bar exit, but does NOT break ties when
entry_long and entry_short fire together. On real data this will manifest
as undefined behaviour — likely whichever engine's branch ran last wins.

REQUIRED PROMOTION GATE
-----------------------
1. Add an explicit conflict-resolution rule (e.g. fade wins in calm
   contango, momentum wins elsewhere) and unit-test it.
2. Phase-15 honest holdout: each engine standalone PLUS the combined
   strategy. Reject promotion if combined Sharpe < max(individual Sharpes).
3. Same numerical floors as the other untested strategies.
"""

STATUS = "UNTESTED"
SUBSTATUS = "NO-REAL-DATA-OOS"
SOURCE_MODULE = "apex.strategies.institutional_arbitrage_engine_v2"
DEPLOYABLE = False


def make_strategy(*_args, **_kwargs):
    raise RuntimeError(
        "institutional_arbitrage_engine_v2 has no validated parameters. "
        "See manifest docstring for the promotion gate."
    )
