"""Abstract base class that every strategy in the ensemble must implement."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class StrategyBase(ABC):
    """Each strategy in the ensemble subclasses this.

    Subclasses MUST set:
      name: str               — short identifier (e.g., "vrp_gex_fade")
      data_requirements: list — keys the strategy expects in the `data` dict
                                passed to compute_signals (e.g., "exec_df_1H",
                                "options_chain_daily", "vix").

    Subclasses MUST implement:
      compute_signals(data) -> DataFrame with columns:
          entry_long, entry_short, exit_long, exit_short (all bool/int Series).
      compute_position_size(data, signals) -> Series of position sizes
          in [-1.0, +1.0]. The ensemble combiner scales these via risk parity.
      get_tunable_params() -> dict[param_name, (lo, hi, type)]
          Optuna search space for this strategy's tunable parameters.
    """

    name: str = ""
    data_requirements: List[str] = []

    @abstractmethod
    def compute_signals(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Return DataFrame with bool columns: entry_long/short, exit_long/short."""

    @abstractmethod
    def compute_position_size(self, data: Dict[str, Any],
                              signals: pd.DataFrame) -> pd.Series:
        """Return per-bar position size in [-1.0, +1.0]."""

    @abstractmethod
    def get_tunable_params(self) -> Dict[str, tuple]:
        """Optuna search space. Returns {param: (lo, hi, type)}.
        type is 'int', 'float', or 'categorical' (with options as 3rd tuple element)."""
