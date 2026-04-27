"""opex_gravity — BLOCKED on options-chain ingest cost.

Status:        BLOCKED-ON-DATA
Source:        apex/strategies/opex_gravity.py

WHY BLOCKED
-----------
Trades the OPEX-week pin-gravity effect (price drifting toward max-OI strike
on monthly expiry weeks). Needs the same per-bar options chain as
vrp_gex_fade. Without the ~200hr Polygon ingest the strategy emits ~zero
signals.

REQUIRED PROMOTION GATE
-----------------------
1. Run the options-chain ingest (or accept the API budget).
2. Set `options_gex.enabled = true`.
3. Phase-15 honest holdout. Sharpe > 1 OOS, n_trades >= 30 (note: only
   ~12 OPEX weeks per year, so trade count will be naturally low — relax
   threshold to >= 12 if needed but penalise heavily in DSR).
"""

STATUS = "BLOCKED-ON-DATA"
SUBSTATUS = "OPTIONS-CHAIN-MISSING"
SOURCE_MODULE = "apex.strategies.opex_gravity"
DEPLOYABLE = False
INGEST_COST_HOURS = 200


def make_strategy(*_args, **_kwargs):
    raise RuntimeError(
        "opex_gravity requires the ~200hr Polygon options-chain ingest. "
        "See manifest docstring for the promotion gate."
    )
