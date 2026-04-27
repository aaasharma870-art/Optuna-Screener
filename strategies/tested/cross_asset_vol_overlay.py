"""Cross-Asset Vol Overlay — TESTED as ensemble component, NOT standalone.

Status:           PASS-AS-OVERLAY (validated as size multiplier in the Phase
                  15/16 ensemble run that produced the Sharpe 3.27 holdout)
Validated at:     2026-04-26
Pipeline version: Phase 16

HONEST CAVEAT
-------------
The overlay's marginal contribution to holdout Sharpe was small — its
multiplier sat near 1.0 for most of the validation window. It is included
in the production preset because it is harmless when the cross-asset signal
is quiet and provides downside-aware sizing when vol regimes diverge. It
should NOT be deployed standalone; it has no entries of its own.

USAGE
-----
Loaded automatically when running with the production preset:
    python apex.py --ensemble --config configs/production/vix_term_v1_config.json
"""

from typing import Any, Dict, Optional


STATUS = "TESTED-AS-OVERLAY"
ROLE = "size_multiplier"
STANDALONE_DEPLOYABLE = False


def make_strategy(params: Optional[Dict[str, Any]] = None):
    """Return a CrossAssetVolOverlayStrategy instance."""
    from apex.strategies.cross_asset_vol_overlay import CrossAssetVolOverlayStrategy
    return CrossAssetVolOverlayStrategy(params=params or {})
