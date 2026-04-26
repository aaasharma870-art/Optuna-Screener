"""Tests for StrategyBase abstract interface."""
import pytest


def test_strategybase_cannot_be_instantiated_directly():
    from apex.strategies.base import StrategyBase
    with pytest.raises(TypeError):
        StrategyBase()


def test_concrete_strategy_must_implement_compute_signals():
    from apex.strategies.base import StrategyBase

    class IncompleteStrategy(StrategyBase):
        name = "incomplete"
        data_requirements = []
        # missing compute_signals
    with pytest.raises(TypeError):
        IncompleteStrategy()


def test_concrete_strategy_works_when_complete():
    from apex.strategies.base import StrategyBase
    import pandas as pd

    class MyStrategy(StrategyBase):
        name = "test_my_strategy"
        data_requirements = ["exec_df_1H"]

        def compute_signals(self, data):
            n = len(data["exec_df_1H"])
            return pd.DataFrame({
                "entry_long": [False] * n,
                "entry_short": [False] * n,
                "exit_long": [False] * n,
                "exit_short": [False] * n,
            })

        def compute_position_size(self, data, signals):
            return pd.Series([0.0] * len(signals))

        def get_tunable_params(self):
            return {}

    s = MyStrategy()
    assert s.name == "test_my_strategy"
    assert s.data_requirements == ["exec_df_1H"]


def test_register_strategy_decorator_adds_to_registry():
    from apex.strategies import STRATEGY_REGISTRY, register_strategy
    from apex.strategies.base import StrategyBase
    import pandas as pd

    @register_strategy
    class RegisteredStrategy(StrategyBase):
        name = "test_registered_strategy"
        data_requirements = []

        def compute_signals(self, data):
            return pd.DataFrame()

        def compute_position_size(self, data, signals):
            return pd.Series()

        def get_tunable_params(self):
            return {}

    assert "test_registered_strategy" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["test_registered_strategy"] is RegisteredStrategy
