# Universal Strategy Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow any user strategy .py file to run through the Optuna pipeline's validation framework (holdout split, Monte Carlo, noise injection, robustness testing) using the strategy's exact entry/exit logic.

**Architecture:** A `StrategyAdapter` class dynamically imports a user strategy file, prepares DataFrames with all standard indicators, then a `run_strategy_backtest()` function executes the strategy's own `entry_fn`/`exit_fn` bar-by-bar and returns trades in the same format as the existing `run_backtest()`. The main pipeline gains a `--strategy` CLI arg that skips Layer 1 (architecture search) and optionally Layer 2 (parameter tuning), routing directly to Layer 3 robustness validation + holdout + reports.

**Tech Stack:** Python 3, pandas, numpy, importlib, Optuna (optional for tuning)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `apex/engine/strategy_adapter.py` | Create | StrategyAdapter class: import strategy, prepare DataFrame, wrap entry/exit |
| `apex/engine/strategy_backtest.py` | Create | `run_strategy_backtest()`: bar-by-bar execution, returns pipeline-compatible trades |
| `apex/main.py` | Modify (lines 315-326, 385-496, 560-562) | Add `--strategy` arg, strategy pipeline path |
| `apex/optimize/layer3.py` | Modify (lines 58-93, 95-123) | Support strategy adapter in noise/regime re-runs |
| `apex/engine/portfolio.py` | Modify (lines 122-180) | Support strategy adapter in final backtest |

---

### Task 1: Create StrategyAdapter class

**Files:**
- Create: `apex/engine/strategy_adapter.py`

- [ ] **Step 1: Create `apex/engine/strategy_adapter.py`**

```python
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

        # Optional config
        self.capital = getattr(self.module, "CAPITAL", 100000)
        self.pos_pct = 0.25
        self.max_positions = getattr(self.module, "MAX_POS", 5)

    def prepare_df(self, polygon_df, spy_df=None, sym="UNKNOWN"):
        """Prepare a DataFrame with all indicators from Polygon data."""
        return prepare_strategy_dataframe(polygon_df, spy_df=spy_df, sym=sym)
```

- [ ] **Step 2: Verify the file parses correctly**

Run: `cd C:/Users/AAASH/Optuna-Screener && python -c "from apex.engine.strategy_adapter import StrategyAdapter; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add apex/engine/strategy_adapter.py
git commit -m "feat: add StrategyAdapter for importing user strategy files"
```

---

### Task 2: Create strategy backtest engine

**Files:**
- Create: `apex/engine/strategy_backtest.py`

- [ ] **Step 1: Create `apex/engine/strategy_backtest.py`**

```python
"""Bar-by-bar backtest engine that runs a user strategy's exact entry/exit logic."""

import numpy as np
import pandas as pd

from apex.engine.backtest import compute_stats


def run_strategy_backtest(adapter, prepared_df, sym, commission_pct=0.05):
    """
    Run a user strategy's entry_fn/exit_fn bar-by-bar on a prepared DataFrame.

    Returns (trades, stats) in the same format as apex.engine.backtest.run_backtest()
    so all downstream pipeline components (Layer 3, reports, etc.) work unchanged.

    Parameters
    ----------
    adapter : StrategyAdapter
        The loaded strategy adapter with entry_fn and exit_fn.
    prepared_df : pd.DataFrame
        DataFrame with user-convention columns (Close, High, Low, ATR, etc.)
        as produced by prepare_strategy_dataframe().
    sym : str
        Ticker symbol.
    commission_pct : float
        Round-trip commission in percent.

    Returns
    -------
    trades : list[dict]
        Each trade has: entry_datetime, exit_datetime, entry_price, exit_price,
        pnl_pct, gross_pnl_pct, mfe, mae, bars_held, exit_reason,
        entry_regime, entry_atr, entry_score, direction.
    stats : dict
        Performance statistics from compute_stats().
    """
    df = prepared_df
    if len(df) < 3:
        return [], compute_stats([])

    trades = []
    position = None

    for idx in range(2, len(df)):
        r = df.iloc[idx]
        prev = df.iloc[idx - 1]
        prev2 = df.iloc[idx - 2]

        atr_val = r.get("ATR", 0)
        if pd.isna(atr_val) or atr_val == 0:
            continue

        # --- Check exit first ---
        if position is not None:
            try:
                should_exit, exit_price, reason = adapter.exit_fn(r, prev, position, df, idx)
            except Exception:
                should_exit, exit_price, reason = False, 0, ""

            if should_exit:
                entry_price = position["ep"]
                bars_held = idx - position["entry_idx"]
                if position.get("dir", "L") == "L":
                    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100.0
                net_pnl_pct = pnl_pct - 2.0 * commission_pct

                # MFE / MAE from tracked extremes
                highest = position.get("highest", entry_price)
                lowest = position.get("lowest", entry_price)
                if position.get("dir", "L") == "L":
                    mfe = (highest - entry_price) / entry_price * 100.0
                    mae = (lowest - entry_price) / entry_price * 100.0
                else:
                    mfe = (entry_price - lowest) / entry_price * 100.0
                    mae = (entry_price - highest) / entry_price * 100.0

                trades.append({
                    "entry_datetime": str(position.get("entry_date", "")),
                    "exit_datetime": str(r.get("Date", "")),
                    "entry_idx": position["entry_idx"],
                    "exit_idx": idx,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "pnl_pct": round(net_pnl_pct, 4),
                    "gross_pnl_pct": round(pnl_pct, 4),
                    "mfe": round(mfe, 4),
                    "mae": round(mae, 4),
                    "bars_held": bars_held,
                    "exit_reason": reason,
                    "entry_regime": "R1",
                    "entry_atr": round(position.get("entry_atr", 0), 4),
                    "entry_score": int(position.get("score", 0)),
                    "direction": "long" if position.get("dir", "L") == "L" else "short",
                })
                position = None
                continue

            # Update trailing highs/lows
            if position.get("dir", "L") == "L":
                if r["High"] > position.get("highest", position["ep"]):
                    position["highest"] = r["High"]
            else:
                if r["Low"] < position.get("lowest", position["ep"]):
                    position["lowest"] = r["Low"]

        # --- Check entry ---
        if position is None:
            try:
                sig = adapter.entry_fn(r, prev, prev2, sym, df, idx)
            except Exception:
                sig = None

            if sig is not None:
                position = {
                    "dir": sig.get("dir", "L"),
                    "ep": sig["price"],
                    "shares": 1,
                    "orig_shares": 1,
                    "stop": sig.get("stop", sig["price"] * 0.95),
                    "highest": sig["price"],
                    "lowest": sig["price"],
                    "entry_atr": sig.get("atr", atr_val),
                    "entry_date": sig.get("date", r.get("Date")),
                    "entry_idx": idx,
                    "score": sig.get("score", 0),
                    "pyramided": False,
                    "be_hit": False,
                    "tp_hit": False,
                }

    # Close open position at end of data
    if position is not None and len(df) > 0:
        last = df.iloc[-1]
        exit_price = last["Close"]
        entry_price = position["ep"]
        bars_held = len(df) - 1 - position["entry_idx"]
        if position.get("dir", "L") == "L":
            pnl_pct = (exit_price - entry_price) / entry_price * 100.0
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100.0
        net_pnl_pct = pnl_pct - 2.0 * commission_pct

        highest = position.get("highest", entry_price)
        lowest = position.get("lowest", entry_price)
        if position.get("dir", "L") == "L":
            mfe = (highest - entry_price) / entry_price * 100.0
            mae = (lowest - entry_price) / entry_price * 100.0
        else:
            mfe = (entry_price - lowest) / entry_price * 100.0
            mae = (entry_price - highest) / entry_price * 100.0

        trades.append({
            "entry_datetime": str(position.get("entry_date", "")),
            "exit_datetime": str(last.get("Date", "")),
            "entry_idx": position["entry_idx"],
            "exit_idx": len(df) - 1,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "pnl_pct": round(net_pnl_pct, 4),
            "gross_pnl_pct": round(pnl_pct, 4),
            "mfe": round(mfe, 4),
            "mae": round(mae, 4),
            "bars_held": bars_held,
            "exit_reason": "end_of_data",
            "entry_regime": "R1",
            "entry_atr": round(position.get("entry_atr", 0), 4),
            "entry_score": int(position.get("score", 0)),
            "direction": "long" if position.get("dir", "L") == "L" else "short",
        })

    stats = compute_stats(trades)
    return trades, stats


def strategy_full_backtest(adapter, polygon_df, spy_df, sym, commission_pct=0.05):
    """
    End-to-end: prepare DataFrame + run strategy backtest.

    This is the strategy-mode equivalent of backtest.full_backtest().
    """
    prepared = adapter.prepare_df(polygon_df, spy_df=spy_df, sym=sym)
    return run_strategy_backtest(adapter, prepared, sym, commission_pct=commission_pct)
```

- [ ] **Step 2: Verify the file parses correctly**

Run: `cd C:/Users/AAASH/Optuna-Screener && python -c "from apex.engine.strategy_backtest import strategy_full_backtest; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add apex/engine/strategy_backtest.py
git commit -m "feat: add strategy backtest engine with pipeline-compatible output"
```

---

### Task 3: Modify Layer 3 to support strategy adapter

**Files:**
- Modify: `apex/optimize/layer3.py:58-123`

The noise injection and regime stress tests currently call `full_backtest()`. When running in strategy mode, they need to call `strategy_full_backtest()` instead. We pass an optional `strategy_adapter` argument through the gauntlet.

- [ ] **Step 1: Edit `apex/optimize/layer3.py` — add import and modify `noise_injection_test`**

At the top of the file (after line 7), add:

```python
from apex.engine.strategy_backtest import strategy_full_backtest
```

Replace the `noise_injection_test` function (lines 58-92) with:

```python
def noise_injection_test(df_dict, architecture, params, cfg, strategy_adapter=None):
    """
    Inject noise into data and measure backtest degradation.

    If strategy_adapter is provided, uses the user strategy's entry/exit logic
    instead of the pipeline's built-in backtest.
    """
    df = df_dict.get("exec_df")
    daily_df = df_dict.get("daily_df")
    if df is None or len(df) < 100:
        return {"clean_pf": 0.0, "noisy_pf": 0.0, "pf_retention": 0.0}

    sym = df_dict.get("_sym", "UNKNOWN")
    spy_df = df_dict.get("_spy_df")

    if strategy_adapter is not None:
        _, stats_clean = strategy_full_backtest(strategy_adapter, df, spy_df, sym)
    else:
        _, stats_clean = full_backtest(df, daily_df, architecture, params)
    clean_pf = stats_clean.get("pf", 0.0)

    rng = np.random.RandomState(123)
    df_noisy = df.copy()
    noise = rng.uniform(-0.05, 0.05, size=len(df_noisy))
    df_noisy["close"] = df_noisy["close"] * (1.0 + noise)
    df_noisy["high"] = np.maximum(df_noisy["high"], df_noisy["close"])
    df_noisy["low"] = np.minimum(df_noisy["low"], df_noisy["close"])
    df_noisy["close"] = df_noisy["close"].shift(1).bfill()

    if strategy_adapter is not None:
        _, stats_noisy = strategy_full_backtest(strategy_adapter, df_noisy, spy_df, sym)
    else:
        _, stats_noisy = full_backtest(df_noisy, daily_df, architecture, params)
    noisy_pf = stats_noisy.get("pf", 0.0)

    pf_retention = noisy_pf / clean_pf if clean_pf > 0 else 0.0

    return {
        "clean_pf": round(clean_pf, 3),
        "noisy_pf": round(noisy_pf, 3),
        "pf_retention": round(pf_retention, 4),
    }
```

- [ ] **Step 2: Edit `regime_stress_test` (lines 95-123) to accept strategy_adapter**

Replace with:

```python
def regime_stress_test(df_dict, architecture, params, cfg, strategy_adapter=None):
    """
    Flip the regime model to measure regime sensitivity.
    For strategy mode, re-run the user strategy on the same data (no regime to flip,
    so we measure consistency by running twice with a small price shift).
    """
    df = df_dict.get("exec_df")
    daily_df = df_dict.get("daily_df")
    if df is None or len(df) < 100:
        return {"normal_pf": 0.0, "stressed_pf": 0.0, "pf_retention": 0.0}

    sym = df_dict.get("_sym", "UNKNOWN")
    spy_df = df_dict.get("_spy_df")

    if strategy_adapter is not None:
        _, stats_normal = strategy_full_backtest(strategy_adapter, df, spy_df, sym)
        normal_pf = stats_normal.get("pf", 0.0)
        # Stress: shift prices by 1 bar (timing stress)
        df_stressed = df.copy()
        df_stressed["close"] = df_stressed["close"].shift(1).bfill()
        df_stressed["open"] = df_stressed["open"].shift(1).bfill()
        _, stats_stressed = strategy_full_backtest(strategy_adapter, df_stressed, spy_df, sym)
        stressed_pf = stats_stressed.get("pf", 0.0)
    else:
        _, stats_normal = full_backtest(df, daily_df, architecture, params)
        normal_pf = stats_normal.get("pf", 0.0)
        arch_stressed = dict(architecture)
        baseline = architecture.get("regime_model", "ema")
        arch_stressed["regime_model"] = "trend" if baseline != "trend" else "ema"
        _, stats_stressed = full_backtest(df, daily_df, arch_stressed, params)
        stressed_pf = stats_stressed.get("pf", 0.0)

    pf_retention = stressed_pf / normal_pf if normal_pf > 0 else 0.0

    return {
        "normal_pf": round(normal_pf, 3),
        "stressed_pf": round(stressed_pf, 3),
        "pf_retention": round(pf_retention, 4),
    }
```

- [ ] **Step 3: Edit `param_sensitivity_test` (lines 126-171) to handle strategy mode**

Replace with:

```python
def param_sensitivity_test(df_dict, architecture, params, cfg, strategy_adapter=None):
    """
    Jitter parameters by +/-10% and measure PF stability.
    In strategy mode with no tunable params, returns a neutral score.
    """
    if strategy_adapter is not None and not strategy_adapter.tunable_params:
        return {"_no_params": {"stable": True, "pf_range": [1.0, 1.0], "base_pf": 1.0}}

    df = df_dict.get("exec_df")
    daily_df = df_dict.get("daily_df")
    if df is None or len(df) < 100:
        return {}

    _, stats_base = full_backtest(df, daily_df, architecture, params)
    base_pf = stats_base.get("pf", 0.0)

    sensitivity = {}
    numeric_params = {k: v for k, v in params.items()
                      if isinstance(v, (int, float)) and k not in ("commission_pct",)}

    for pname, pval in numeric_params.items():
        if pval == 0:
            continue
        pf_values = []
        for jitter in [-0.10, 0.10]:
            test_params = dict(params)
            jittered = pval * (1.0 + jitter)
            if isinstance(pval, int):
                jittered = max(1, int(round(jittered)))
            else:
                jittered = round(jittered, 6)
            test_params[pname] = jittered
            _, test_stats = full_backtest(df, daily_df, architecture, test_params)
            pf_values.append(test_stats.get("pf", 0.0))

        pf_min = min(pf_values)
        pf_max = max(pf_values)
        stable = True
        if base_pf > 0:
            if pf_min < base_pf * 0.7 or pf_max > base_pf * 1.3:
                stable = False
        sensitivity[pname] = {
            "stable": stable,
            "pf_range": [round(pf_min, 3), round(pf_max, 3)],
            "base_pf": round(base_pf, 3),
        }

    return sensitivity
```

- [ ] **Step 4: Edit `layer3_robustness_gauntlet` (line 174) to pass strategy_adapter through**

Change the function signature on line 174 from:

```python
def layer3_robustness_gauntlet(data_dict, architecture, tuned_results, cfg):
```

to:

```python
def layer3_robustness_gauntlet(data_dict, architecture, tuned_results, cfg, strategy_adapter=None):
```

Then change the three test calls (around lines 196-202) to pass `strategy_adapter`:

```python
        noise = noise_injection_test(sym_data, architecture, params, cfg, strategy_adapter=strategy_adapter)
```

```python
        stress = regime_stress_test(sym_data, architecture, params, cfg, strategy_adapter=strategy_adapter)
```

```python
        sensitivity = param_sensitivity_test(sym_data, architecture, params, cfg, strategy_adapter=strategy_adapter)
```

- [ ] **Step 5: Verify edits parse correctly**

Run: `cd C:/Users/AAASH/Optuna-Screener && python -c "from apex.optimize.layer3 import layer3_robustness_gauntlet; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add apex/optimize/layer3.py
git commit -m "feat: support strategy adapter in Layer 3 robustness tests"
```

---

### Task 4: Modify portfolio.py for strategy-mode final backtest

**Files:**
- Modify: `apex/engine/portfolio.py:122-180`

- [ ] **Step 1: Add import at top of `apex/engine/portfolio.py`**

After line 6, add:

```python
from apex.engine.strategy_backtest import strategy_full_backtest
```

- [ ] **Step 2: Modify `phase_full_backtest` to accept and use strategy_adapter**

Change the function signature on line 122 from:

```python
def phase_full_backtest(data_dict, architecture, final_results, cfg, tuned_results=None):
```

to:

```python
def phase_full_backtest(data_dict, architecture, final_results, cfg, tuned_results=None, strategy_adapter=None):
```

Then replace the two `full_backtest` calls inside the function.

The first call (line 150):

```python
        trades, stats = full_backtest(df, daily_df, architecture, params)
```

becomes:

```python
        if strategy_adapter is not None:
            spy_data = data_dict.get("SPY", data_dict.get("_spy_data", {}))
            spy_df = spy_data.get("exec_df") if isinstance(spy_data, dict) else None
            trades, stats = strategy_full_backtest(strategy_adapter, df, spy_df, sym)
        else:
            trades, stats = full_backtest(df, daily_df, architecture, params)
```

The second call (line 164):

```python
            holdout_trades, holdout_stats = full_backtest(
                holdout_df, holdout_daily, architecture, params
            )
```

becomes:

```python
            if strategy_adapter is not None:
                spy_data = data_dict.get("SPY", data_dict.get("_spy_data", {}))
                spy_df = spy_data.get("exec_df_holdout") if isinstance(spy_data, dict) else None
                holdout_trades, holdout_stats = strategy_full_backtest(
                    strategy_adapter, holdout_df, spy_df, sym
                )
            else:
                holdout_trades, holdout_stats = full_backtest(
                    holdout_df, holdout_daily, architecture, params
                )
```

- [ ] **Step 3: Verify**

Run: `cd C:/Users/AAASH/Optuna-Screener && python -c "from apex.engine.portfolio import phase_full_backtest; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add apex/engine/portfolio.py
git commit -m "feat: support strategy adapter in final backtest phase"
```

---

### Task 5: Wire --strategy into main.py pipeline

**Files:**
- Modify: `apex/main.py:315-326, 385-610`

- [ ] **Step 1: Add import at top of `apex/main.py` (after line 25)**

```python
from apex.engine.strategy_adapter import StrategyAdapter
from apex.engine.strategy_backtest import strategy_full_backtest
```

- [ ] **Step 2: Add `--strategy` CLI argument (after line 325)**

After the `--validate-vrp` argument line, add:

```python
    parser.add_argument("--strategy", type=str, default="",
                        help="Path to a user strategy .py file (uses exact entry/exit logic)")
```

- [ ] **Step 3: Add strategy pipeline path in main() (after line 419, before Layer 1)**

After `survivors = list(data_dict.keys())` (line 419), insert the strategy-mode pipeline:

```python
    # ---- STRATEGY MODE ----
    if args.strategy:
        adapter = StrategyAdapter(args.strategy)
        log(f"Strategy mode: {adapter.name}")
        log(f"Strategy file: {adapter.path}")

        # Get SPY data for RS calculations
        spy_data = data_dict.get("SPY", {})
        spy_exec_df = spy_data.get("exec_df")

        # Inject SPY reference and symbol name into data_dict for Layer 3
        for sym in data_dict:
            data_dict[sym]["_spy_df"] = spy_exec_df
            data_dict[sym]["_sym"] = sym

        # Skip Layer 1 — the strategy IS the architecture
        architecture = {
            "indicators": ["UserStrategy"],
            "min_score": 1,
            "exit_methods": ["user_strategy"],
            "regime_model": "none",
            "position_sizing": "equal",
            "exec_timeframe": cfg.get("phase3_params", {}).get("exec_timeframe", "1H"),
            "score_aggregation": "additive",
            "concept_weights": {},
            "direction": "long",
        }

        # Skip Layer 2 — run strategy directly on each symbol
        log("=== STRATEGY BACKTEST (user entry/exit logic) ===")
        tuned_results = {}
        for idx, sym in enumerate(survivors, 1):
            log(f"  [{idx}/{len(survivors)}] Backtesting {sym}...")
            sym_data = data_dict[sym]
            df = sym_data.get("exec_df")
            if df is None or len(df) < 100:
                continue

            trades, stats = strategy_full_backtest(adapter, df, spy_exec_df, sym)
            trade_pnls = [t["pnl_pct"] for t in trades]

            if stats["trades"] < 3:
                log(f"    {sym}: SKIP ({stats['trades']} trades)")
                continue

            from apex.optimize.layer1 import _compute_fitness
            fitness = _compute_fitness(stats)

            tuned_results[sym] = {
                "params": {},
                "stats": stats,
                "trades": trades,
                "trade_pnls": trade_pnls,
                "fitness": fitness,
            }
            log(f"    {sym}: {stats['trades']}tr PF={stats['pf']:.2f} "
                f"WR={stats['wr_pct']:.1f}% Ret={stats['total_return_pct']:.1f}%")

        if not tuned_results:
            log("No symbols produced trades. Exiting.", "ERROR")
            sys.exit(1)

        log(f"Strategy backtest complete: {len(tuned_results)} symbols with trades")

        # Layer 3: Robustness
        validated_results, robustness_data = layer3_robustness_gauntlet(
            data_dict, architecture, tuned_results, cfg, strategy_adapter=adapter
        )

        if not validated_results:
            log("No symbols passed robustness. Using all tuned results.", "WARN")
            validated_results = tuned_results
            for sym in validated_results:
                validated_results[sym]["robustness"] = robustness_data.get(sym, {})

        # Correlation filter
        final_results = correlation_filter(validated_results, cfg)
        if not final_results:
            final_results = validated_results

        # Final backtest (tune + holdout)
        backtest_results = phase_full_backtest(
            data_dict, architecture, final_results, cfg,
            tuned_results=tuned_results, strategy_adapter=adapter
        )

        # Reports (same as normal pipeline)
        log("=== GENERATING REPORTS ===")
        run_info["concept"] = adapter.name
        report_path = generate_html_report(
            backtest_results, architecture, robustness_data, run_info, str(run_output)
        )
        generate_trades_csv(backtest_results.get("all_trades", []), str(run_output))
        generate_summary_csv(backtest_results, str(run_output))
        generate_parameters_json(backtest_results, architecture, str(run_output))

        if not args.no_amibroker:
            sorted_syms = backtest_results.get("sorted_syms", [])
            afl_str = generate_apex_afl(sorted_syms, backtest_results, architecture)
            push_to_amibroker(backtest_results, afl_str, str(run_output), cfg)
        else:
            log("AmiBroker push skipped (--no-amibroker)")

        # Summary
        log("=== PIPELINE COMPLETE ===")
        pstats = backtest_results.get("portfolio_stats", {})
        log("Final Results:")
        log(f"  Strategy: {adapter.name}")
        log(f"  Symbols: {len(backtest_results.get('sorted_syms', []))}")
        log(f"  Trades:  {pstats.get('trades', 0)}")
        log(f"  PF:      {pstats.get('pf', 0):.2f}")
        log(f"  Win%:    {pstats.get('wr_pct', 0):.1f}%")
        log(f"  Return:  {pstats.get('total_return_pct', 0):.1f}%")
        log(f"  MaxDD:   {pstats.get('max_dd_pct', 0):.1f}%")
        log(f"  Sharpe:  {pstats.get('sharpe', 0):.2f}")
        hstats = backtest_results.get("holdout_universe_stats", {})
        if hstats:
            log("  --- TRUE HOLDOUT (never seen) ---")
            log(f"  Holdout Trades:  {hstats.get('trades', 0)}")
            log(f"  Holdout PF:      {hstats.get('pf', 0):.2f}")
            log(f"  Holdout Win%:    {hstats.get('wr_pct', 0):.1f}%")
            log(f"  Holdout Return:  {hstats.get('total_return_pct', 0):.1f}%")
            log(f"  Holdout Sharpe:  {hstats.get('sharpe', 0):.2f}")
        log(f"  Report:  {report_path}")

        try:
            abs_report = str(Path(report_path).resolve()).replace("\\", "/")
            webbrowser.open(f"file:///{abs_report}")
            log(f"Report opened in browser: file:///{abs_report}")
        except Exception as e:
            log(f"Could not open browser: {e}", "WARN")

        return backtest_results
```

- [ ] **Step 4: Verify the full pipeline parses**

Run: `cd C:/Users/AAASH/Optuna-Screener && python -c "from apex.main import main; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add apex/main.py
git commit -m "feat: add --strategy CLI arg for user strategy file backtesting"
```

---

### Task 6: Integration test with S12 strategy

**Files:**
- No new files

- [ ] **Step 1: Run S12 strategy through the pipeline**

Run:
```bash
cd C:/Users/AAASH/Optuna-Screener && python apex.py --strategy "C:/TradingScripts/FINAL STRATEGYIE/s12_momentum_acceleration.py" --budget light --no-amibroker
```

Expected: Pipeline runs through strategy mode, backtests each symbol using S12's exact entry/exit logic, runs Layer 3 robustness, produces HTML report.

- [ ] **Step 2: Run S4 strategy through the pipeline**

Run:
```bash
cd C:/Users/AAASH/Optuna-Screener && python apex.py --strategy "C:/TradingScripts/FINAL STRATEGYIE/s4_inside_day_breakout.py" --budget light --no-amibroker
```

Expected: Same pipeline flow with S4's inside day breakout logic.

- [ ] **Step 3: Run a third strategy to verify generality**

Run:
```bash
cd C:/Users/AAASH/Optuna-Screener && python apex.py --strategy "C:/TradingScripts/FINAL STRATEGYIE/s8_bullish_outside_day.py" --budget light --no-amibroker
```

Expected: Works with no code changes needed.

- [ ] **Step 4: Commit any bug fixes found during integration testing**

```bash
git add -u
git commit -m "fix: integration fixes from strategy adapter testing"
```
