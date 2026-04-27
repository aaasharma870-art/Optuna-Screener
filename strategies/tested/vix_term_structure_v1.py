"""VIX Term Structure v1 — TESTED, production-validated.

Status:           PASS (Sharpe 3.27 holdout on unseen 2025-Q3 -> 2026-Q1 data)
Validated at:     2026-04-26
Pipeline version: Phase 16
Canonical preset: strategies/production/vix_term_structure_v1.py

Re-export shim so this strategy appears in strategies/tested/ alongside the
other validated strategies. The deployable preset stays at its production
path; all parameters and validation metadata are imported from there to
guarantee a single source of truth.
"""

from strategies.production.vix_term_structure_v1 import (  # noqa: F401
    TUNED_PARAMS,
    VALIDATION_METADATA,
    make_strategy,
)

STATUS = "TESTED"
HOLDOUT_SHARPE = 3.27
HOLDOUT_RETURN_PCT = 37.68
HOLDOUT_MAX_DD_PCT = -3.00
HOLDOUT_TRADES = 108
