"""vrp_gex_fade — BLOCKED on options-chain ingest cost.

Status:        BLOCKED-ON-DATA
Source:        apex/strategies/vrp_gex_fade.py
Phase 14 wire: apex/data/options_gex.py (synthetic greeks via Black-Scholes)

WHY BLOCKED
-----------
The strategy fades GEX (gamma exposure) imbalances, which requires a per-bar
option chain across the SPY/QQQ universe. Phase 14 wired the Polygon options
contract + price fetcher and a Black-Scholes synthetic-greeks fallback, but
running the full historical chain ingest costs ~200 hours of Polygon Stocks
+ Options Starter API time at the documented rate limit. The full ingest
has not been performed, so this strategy emits ~zero signals when run with
the default `options_gex.enabled = false`.

REQUIRED PROMOTION GATE
-----------------------
1. Decide whether to budget the ~200hr ingest (or upgrade to a higher tier).
2. Set `options_gex.enabled = true` in apex_config.json.
3. Run Phase-15 honest holdout on real GEX-driven signals.
4. Sharpe > 1 OOS, MaxDD > -10%, n_trades >= 30.
"""

STATUS = "BLOCKED-ON-DATA"
SUBSTATUS = "OPTIONS-CHAIN-MISSING"
SOURCE_MODULE = "apex.strategies.vrp_gex_fade"
DEPLOYABLE = False
INGEST_COST_HOURS = 200


def make_strategy(*_args, **_kwargs):
    raise RuntimeError(
        "vrp_gex_fade requires the ~200hr Polygon options-chain ingest. "
        "See manifest docstring for the promotion gate."
    )
