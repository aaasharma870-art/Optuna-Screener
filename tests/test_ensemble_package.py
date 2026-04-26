"""Sanity check: apex.ensemble package importable."""
def test_ensemble_package_importable():
    import apex.ensemble
    assert apex.ensemble.__name__ == "apex.ensemble"
