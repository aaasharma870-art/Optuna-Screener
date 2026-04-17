"""Regime-specific fitness functions for multi-objective optimization."""

import math

SUPPRESSED_REGIMES = {"R1", "R2", "Contango_Calm", "Neutral_Calm"}
AMPLIFIED_REGIMES = {"R3", "Contango_Elevated", "Neutral_Elevated",
                     "Backwardation_Calm", "Backwardation_Elevated"}

MIN_DD_CAP = 0.5
MIN_LOSS_CAP = 0.1


def suppressed_fitness(win_rate_pct, profit_factor):
    """
    Fitness for suppressed regimes: win_rate_pct^2 * profit_factor.

    Returns zero if profit_factor <= 0.
    """
    if profit_factor <= 0:
        return 0.0
    return (win_rate_pct ** 2) * profit_factor


def amplified_fitness(total_return_pct, max_dd_pct, avg_win, avg_loss):
    """
    Fitness for amplified regimes:
    (total_return / max(|dd|, MIN_DD_CAP)) * (avg_win / max(|avg_loss|, MIN_LOSS_CAP))
    """
    dd_denom = max(abs(max_dd_pct), MIN_DD_CAP)
    loss_denom = max(abs(avg_loss), MIN_LOSS_CAP)
    return (total_return_pct / dd_denom) * (avg_win / loss_denom)


def compute_regime_fitness(regime_state, stats):
    """
    Dispatch fitness computation based on regime state.

    - Suppressed regimes -> suppressed_fitness(win_rate, pf)
    - Amplified regimes -> amplified_fitness(return, dd, avg_win, avg_loss)
    - Unknown regimes -> legacy PF * sqrt(trades) * (1 - dd/100)
    """
    if regime_state in SUPPRESSED_REGIMES:
        return suppressed_fitness(
            stats.get("wr_pct", 0.0),
            stats.get("pf", 0.0),
        )

    if regime_state in AMPLIFIED_REGIMES:
        return amplified_fitness(
            stats.get("total_return_pct", 0.0),
            stats.get("max_dd_pct", 0.0),
            stats.get("avg_win", 0.0),
            stats.get("avg_loss", 0.0),
        )

    # Unknown / legacy fallback
    pf = stats.get("pf", 0.0)
    trades = stats.get("trades", 0)
    dd = stats.get("max_dd_pct", 0.0)
    if trades <= 0 or pf <= 0:
        return 0.0
    return pf * math.sqrt(trades) * (1.0 - dd / 100.0)
