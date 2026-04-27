"""vol_skew_arb — BLOCKED on options-chain ingest cost.

Status:        BLOCKED-ON-DATA
Source:        apex/strategies/vol_skew_arb.py

WHY BLOCKED
-----------
Trades dislocations in the implied-vol skew (25-delta put IV vs 25-delta
call IV) versus realised vol. Needs the per-bar options chain. Without the
~200hr Polygon ingest the strategy emits ~zero signals.

REQUIRED PROMOTION GATE
-----------------------
1. Options-chain ingest complete and `options_gex.enabled = true`.
2. Verify skew computation against a known reference (e.g. CBOE SKEW index
   on the same dates) before backtesting — Black-Scholes synthetic greeks
   can drift from market IV in stressed regimes.
3. Phase-15 honest holdout. Sharpe > 1 OOS, MaxDD > -10%, n_trades >= 30.
"""

STATUS = "BLOCKED-ON-DATA"
SUBSTATUS = "OPTIONS-CHAIN-MISSING"
SOURCE_MODULE = "apex.strategies.vol_skew_arb"
DEPLOYABLE = False
INGEST_COST_HOURS = 200


def make_strategy(*_args, **_kwargs):
    raise RuntimeError(
        "vol_skew_arb requires the ~200hr Polygon options-chain ingest. "
        "See manifest docstring for the promotion gate."
    )
