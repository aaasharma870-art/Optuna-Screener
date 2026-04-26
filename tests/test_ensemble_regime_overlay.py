"""Tests for regime-conditional weight tilts."""
import pytest


MEAN_REV_STRATEGIES = ("vrp_gex_fade", "vol_skew_arb", "smc_structural")
TREND_STRATEGIES = ("opex_gravity", "vix_term_structure")


def test_r1_boosts_mean_reversion_strategies():
    from apex.ensemble.regime_overlay import apply_regime_tilts
    base = {"vrp_gex_fade": 0.2, "opex_gravity": 0.2, "vix_term_structure": 0.2,
            "vol_skew_arb": 0.2, "smc_structural": 0.2}
    tilted = apply_regime_tilts(base, regime="R1")
    # Mean-rev strategies should have higher weight than trend strategies
    for mr in MEAN_REV_STRATEGIES:
        for tr in TREND_STRATEGIES:
            assert tilted[mr] > tilted[tr]
    assert sum(tilted.values()) == pytest.approx(1.0, abs=1e-9)


def test_r3_boosts_trend_strategies():
    from apex.ensemble.regime_overlay import apply_regime_tilts
    base = {"vrp_gex_fade": 0.2, "opex_gravity": 0.2, "vix_term_structure": 0.2,
            "vol_skew_arb": 0.2, "smc_structural": 0.2}
    tilted = apply_regime_tilts(base, regime="R3")
    for tr in TREND_STRATEGIES:
        for mr in MEAN_REV_STRATEGIES:
            assert tilted[tr] > tilted[mr]
    assert sum(tilted.values()) == pytest.approx(1.0, abs=1e-9)


def test_r4_zeros_all_weights():
    from apex.ensemble.regime_overlay import apply_regime_tilts
    base = {"vrp_gex_fade": 0.2, "opex_gravity": 0.2, "vix_term_structure": 0.2,
            "vol_skew_arb": 0.2, "smc_structural": 0.2}
    tilted = apply_regime_tilts(base, regime="R4")
    for w in tilted.values():
        assert w == 0.0


def test_unknown_regime_returns_unchanged():
    from apex.ensemble.regime_overlay import apply_regime_tilts
    base = {"vrp_gex_fade": 0.5, "opex_gravity": 0.5}
    tilted = apply_regime_tilts(base, regime="UNKNOWN")
    assert tilted == base


def test_missing_strategy_in_base_is_handled():
    """Tilts should only apply to strategies actually present in the weights."""
    from apex.ensemble.regime_overlay import apply_regime_tilts
    base = {"vrp_gex_fade": 0.5, "opex_gravity": 0.5}
    tilted = apply_regime_tilts(base, regime="R1")
    assert "vrp_gex_fade" in tilted and "opex_gravity" in tilted
    assert sum(tilted.values()) == pytest.approx(1.0, abs=1e-9)
