"""Sanity check: apex.strategies package importable with empty registry."""
def test_strategies_package_importable():
    import apex.strategies
    assert hasattr(apex.strategies, "STRATEGY_REGISTRY")
    assert isinstance(apex.strategies.STRATEGY_REGISTRY, dict)
