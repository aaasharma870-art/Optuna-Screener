"""Regime-conditional weight tilts for the ensemble."""
from typing import Dict


MEAN_REVERSION_STRATEGIES = {"vrp_gex_fade", "vol_skew_arb", "smc_structural"}
TREND_STRATEGIES = {"opex_gravity", "vix_term_structure"}

TILT_FACTOR = 1.20  # 20% boost to favored strategies in current regime


def apply_regime_tilts(weights: Dict[str, float],
                        regime: str) -> Dict[str, float]:
    """Apply regime-conditional multiplicative tilts and renormalize.

    Suppressed regimes (R1, R2, Contango_Calm, Neutral_Calm):
        boost mean-reversion strategies by TILT_FACTOR.
    Amplified regimes (R3, Backwardation, Elevated VRP):
        boost trend-following strategies by TILT_FACTOR.
    R4 (no-trade): all weights → 0.
    Unknown regime: weights returned unchanged.
    """
    if regime == "R4":
        return {name: 0.0 for name in weights}

    if regime in {"R1", "R2", "Contango_Calm", "Neutral_Calm"}:
        boost_set = MEAN_REVERSION_STRATEGIES
    elif regime in {"R3", "Contango_Elevated", "Neutral_Elevated",
                    "Backwardation_Calm", "Backwardation_Elevated"}:
        boost_set = TREND_STRATEGIES
    else:
        return dict(weights)

    tilted = {}
    for name, w in weights.items():
        if name in boost_set:
            tilted[name] = w * TILT_FACTOR
        else:
            tilted[name] = w

    total = sum(tilted.values())
    if total <= 1e-9:
        return tilted
    return {name: w / total for name, w in tilted.items()}
