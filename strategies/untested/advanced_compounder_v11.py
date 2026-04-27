"""advanced_compounder_v11 — UNTESTED (no real-data OOS) + design concern.

Status:        NO-REAL-DATA-OOS
Source:        apex/strategies/advanced_compounder_v11.py (154 LOC)
Added:         2026-04-26 (commit 9e3f95e)

WHY UNTESTED
------------
4 unit tests on synthetic toy bars verify Supertrend computation and
pyramiding cap. No real-data validation.

DESIGN CONCERN
--------------
This is a literal Pine port: macro Supertrend defines regime, micro
Supertrend flips trigger pyramid adds, macro flip closes. There is no VRP /
VIX / term-structure context, no vol filter, no regime overlay. Per the
project's structural-strategy stance ("if it were as easy as indicators
everyone would do it"), pure indicator-based trend-following is unlikely to
clear the Aryan Optimized 77.7% credit-spread DSR baseline on its own.

REQUIRED PROMOTION GATE
-----------------------
Realistically, this strategy needs more than just a holdout pass — it needs
a regime overlay before validation is even worth running. Suggested order:
  1. Add a VIX/VIX3M contango filter (only allow longs when contango).
  2. Add a vol-target sizer instead of fixed 0.20 unit pyramiding.
  3. THEN run Phase-15 holdout. Anything less and the validation result
     will be dominated by 2021-2024 trend regime luck.
"""

STATUS = "UNTESTED"
SUBSTATUS = "NO-REAL-DATA-OOS"
SOURCE_MODULE = "apex.strategies.advanced_compounder_v11"
DEPLOYABLE = False


def make_strategy(*_args, **_kwargs):
    raise RuntimeError(
        "advanced_compounder_v11 has no validated parameters and lacks a "
        "regime overlay. See manifest docstring for the promotion gate."
    )
