"""Universal strategy adapter: import any user strategy .py and run it in the pipeline."""

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_strategy_module(strategy_path):
    """Dynamically import a strategy .py file as a module."""
    path = Path(strategy_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")
    if not path.suffix == ".py":
        raise ValueError(f"Strategy file must be a .py file: {path}")

    module_name = f"user_strategy_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _calc_rsi(close, period=14):
    """RSI computation matching user strategy convention."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _calc_atr(high, low, close, period=14):
    """ATR computation matching user strategy convention."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def prepare_strategy_dataframe(df, spy_df=None, sym="UNKNOWN"):
    """
    Prepare a DataFrame with ALL standard indicator columns that user
    strategies expect.  Works on both daily and intraday OHLCV data from
    the Polygon pipeline (columns: open, high, low, close, volume, datetime).

    Renames Polygon columns to user convention (Close, High, Low, etc.)
    and computes every indicator used across all known user strategies.
    """
    out = pd.DataFrame()
    out["Date"] = df["datetime"]
    out["Open"] = df["open"].values
    out["High"] = df["high"].values
    out["Low"] = df["low"].values
    out["Close"] = df["close"].values
    out["Volume"] = df["volume"].values

    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    # --- Standard indicators (all strategies use these) ---
    out["RSI"] = _calc_rsi(close, 14)
    out["ATR"] = _calc_atr(high, low, close, 14)
    out["ATR_pct"] = out["ATR"] / close * 100

    out["EMA10"] = close.ewm(span=10, adjust=False).mean()
    out["EMA21"] = close.ewm(span=21, adjust=False).mean()
    out["SMA10"] = close.rolling(10).mean()
    out["SMA21"] = close.rolling(21).mean()
    out["SMA50"] = close.rolling(50).mean()
    out["SMA200"] = close.rolling(200).mean()

    out["Vol_MA20"] = out["Volume"].rolling(20).mean()
    out["Vol_Ratio"] = out["Volume"] / out["Vol_MA20"].replace(0, np.nan)
    out["Vol_OK"] = out["Volume"] > out["Vol_MA20"]

    out["Trend_OK"] = (out["SMA10"] > out["SMA21"]) & (out["SMA21"] > out["SMA50"])
    out["Above50"] = close > out["SMA50"]
    out["Above200"] = close > out["SMA200"] if not out["SMA200"].isna().all() else False

    out["High_52w"] = high.rolling(252, min_periods=100).max()
    out["High_20d"] = high.rolling(20).max()
    out["Low_20d"] = low.rolling(20).min()
    out["Range_10d"] = (high.rolling(10).max() - low.rolling(10).min()) / close * 100

    # --- RS vs SPY (21-day relative strength) ---
    out["RS_21d"] = 0.0
    if spy_df is not None and sym != "SPY":
        spy_close = spy_df.set_index("datetime")["close"]
        stock_close = df.set_index("datetime")["close"]
        spy_aligned = spy_close.reindex(stock_close.index, method="ffill")
        stock_ret = stock_close.pct_change(21)
        spy_ret = spy_aligned.pct_change(21)
        rs_vals = (stock_ret.values - spy_ret.values) * 100
        out["RS_21d"] = rs_vals

    # --- S12: Momentum Acceleration ---
    out["Return_5d"] = (close - close.shift(5)) / close.shift(5) * 100
    out["Return_10d"] = (close - close.shift(10)) / close.shift(10) * 100
    out["Momentum_Accel"] = (out["Return_5d"] > out["Return_10d"]) & (
        out["Return_5d"].shift(1) <= out["Return_10d"].shift(1)
    )

    # --- S4: Inside Day ---
    out["Inside_Day"] = (high < high.shift(1)) & (low > low.shift(1))

    # --- S8: Bullish Outside Day ---
    out["Outside_Day"] = (high > high.shift(1)) & (low < low.shift(1))
    out["Bullish_Outside"] = out["Outside_Day"] & (close > out["Open"])

    # --- S2: Pocket Pivot ---
    down_vol = out["Volume"].where(close < close.shift(1), np.nan)
    out["Max_Down_Vol_10"] = down_vol.rolling(10, min_periods=1).max()
    out["Pocket_Pivot"] = (
        (close > close.shift(1))
        & (out["Volume"] > out["Max_Down_Vol_10"].shift(1))
        & (close > out["EMA21"])
    )

    # --- S7: RDZ Momentum ---
    rsi_ma20 = out["RSI"].rolling(20).mean()
    rsi_std20 = out["RSI"].rolling(20).std()
    out["RDZ"] = (out["RSI"] - rsi_ma20) / rsi_std20.replace(0, np.nan)
    price_change = close.pct_change() * 100
    out["Sigma"] = price_change / price_change.rolling(100, min_periods=20).std().replace(0, np.nan)

    # --- S10: RS New Highs ---
    if spy_df is not None and sym != "SPY":
        spy_close_series = spy_df.set_index("datetime")["close"]
        stock_close_series = df.set_index("datetime")["close"]
        spy_al = spy_close_series.reindex(stock_close_series.index, method="ffill")
        rs_line = (stock_close_series / spy_al.replace(0, np.nan)).values
        out["RS_Line"] = rs_line
        out["RS_Line_High_20"] = pd.Series(rs_line).rolling(20, min_periods=10).max().values
        out["RS_New_High_20"] = (
            pd.Series(rs_line).values >= pd.Series(rs_line).rolling(20, min_periods=10).max().values
        ) & (close.values < high.rolling(20).max().values)
    else:
        out["RS_Line"] = 1.0
        out["RS_Line_High_20"] = 1.0
        out["RS_New_High_20"] = False

    return out


class StrategyAdapter:
    """
    Wraps a user strategy .py file for use in the Optuna pipeline.

    The strategy file must define:
      - entry_fn(r, prev, prev2, sym, df, idx) -> dict | None
      - exit_fn(r, prev, pos, df, idx) -> (bool, float, str)

    Optionally:
      - TUNABLE_PARAMS = {"param_name": (min, max), ...}
      - CAPITAL, START, END constants
    """

    def __init__(self, strategy_path):
        self.path = Path(strategy_path).resolve()
        self.module = _load_strategy_module(strategy_path)

        # Validate required functions
        if not hasattr(self.module, "entry_fn"):
            raise AttributeError(f"Strategy {self.path.name} missing entry_fn()")
        if not hasattr(self.module, "exit_fn"):
            raise AttributeError(f"Strategy {self.path.name} missing exit_fn()")

        self.entry_fn = self.module.entry_fn
        self.exit_fn = self.module.exit_fn
        self.name = getattr(self.module, "__doc__", self.path.stem) or self.path.stem
        self.name = self.name.strip().split("\n")[0][:80]

        # Optional tunable params for Optuna Layer 2
        self.tunable_params = getattr(self.module, "TUNABLE_PARAMS", {})

        # Store default params for reset
        self.default_params = dict(getattr(self.module, "PARAMS", {}))

        # Optional config
        self.capital = getattr(self.module, "CAPITAL", 100000)
        self.pos_pct = 0.25
        self.max_positions = getattr(self.module, "MAX_POS", 5)

    def set_params(self, params):
        """Inject tuned parameters into the strategy module's PARAMS dict."""
        if hasattr(self.module, "PARAMS"):
            for k, v in params.items():
                if k in self.module.PARAMS:
                    self.module.PARAMS[k] = v

    def reset_params(self):
        """Reset strategy parameters to defaults."""
        if hasattr(self.module, "PARAMS"):
            self.module.PARAMS.update(self.default_params)

    def prepare_df(self, polygon_df, spy_df=None, sym="UNKNOWN"):
        """Prepare a DataFrame with all indicators from Polygon data."""
        return prepare_strategy_dataframe(polygon_df, spy_df=spy_df, sym=sym)
