"""Verify Strategy 1 is loadable from the registry."""
def test_vrp_gex_fade_in_registry():
    import apex.strategies.vrp_gex_fade  # noqa: F401  triggers registration
    from apex.strategies import STRATEGY_REGISTRY
    assert "vrp_gex_fade" in STRATEGY_REGISTRY
    cls = STRATEGY_REGISTRY["vrp_gex_fade"]
    instance = cls()
    assert instance.name == "vrp_gex_fade"
    assert "exec_df_1H" in instance.data_requirements
