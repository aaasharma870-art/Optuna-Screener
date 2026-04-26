"""Strategy modules for the institutional multi-strategy ensemble.

Each strategy implements StrategyBase (see base.py). Strategies register
themselves via @register_strategy decorator on their class.
"""
STRATEGY_REGISTRY: dict = {}


def register_strategy(cls):
    """Class decorator that registers a strategy in STRATEGY_REGISTRY."""
    if not hasattr(cls, "name"):
        raise TypeError(f"{cls.__name__} must define a class attribute `name`")
    STRATEGY_REGISTRY[cls.name] = cls
    return cls
