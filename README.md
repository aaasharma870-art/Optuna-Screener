# Optuna Screener — Automated Trading Strategy Research Pipeline

> A Python pipeline that discovers, optimizes, and validates systematic trading strategies
> using real market data, Bayesian hyperparameter search, and statistical robustness testing.
> Drop in any strategy file and get an honest assessment of whether it works on unseen data.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Built with Optuna](https://img.shields.io/badge/built%20with-Optuna-8A2BE2)
![Data: Polygon.io](https://img.shields.io/badge/data-Polygon.io-orange)

---

## The Problem

Anyone can write a trading strategy that looks profitable on historical data. The hard
part is knowing whether those results are real or a statistical accident. Most backtest
tools let the user peek at the same data they're optimizing against, which inflates
results and creates strategies that collapse on live markets. This project exists to
answer one question honestly: **does this strategy actually work on data it has never seen?**

## What This Project Does

The pipeline takes a trading strategy — either one you wrote yourself or one it
discovers automatically — and subjects it to a multi-layered validation process designed
to separate genuine edge from noise.

**Two modes of operation:**

### 1. Strategy File Mode (drop-in backtesting)

Point the pipeline at any Python strategy file with `entry_fn` / `exit_fn` functions.
It imports your exact trading logic, runs it bar-by-bar on fresh market data from
Polygon.io, optimizes your tunable parameters with walk-forward validation, stress-tests
with Monte Carlo simulation, and reports honest results on a 25% holdout window the
optimizer never touched.

```bash
python apex.py --strategy my_strategy.py --budget light
```

Your strategy file just needs two functions:

```python
def entry_fn(r, prev, prev2, sym, df, idx):
    """Return a signal dict to enter, or None to skip."""
    if r['RSI'] < 30 and r['Trend_OK'] and r['Vol_OK']:
        return {'sym': sym, 'dir': 'L', 'price': r['Close'],
                'stop': r['Close'] - 1.5 * r['ATR'],
                'atr': r['ATR'], 'date': r['Date'], 'score': 1.0}
    return None

def exit_fn(r, prev, pos, df, idx):
    """Return (should_exit, exit_price, reason)."""
    if r['Low'] <= pos['stop']:
        return True, pos['stop'], 'Trail Stop'
    return False, 0, ''
```

The pipeline automatically computes all standard indicators (RSI, ATR, moving averages,
volume ratios, relative strength vs SPY, and more) so your entry/exit logic has
everything it needs. If you define a `TUNABLE_PARAMS` dictionary, Optuna will search
those parameter ranges with walk-forward IS/OOS optimization.

### 2. Discovery Mode (architecture search)

Give the pipeline a concept in plain English and it searches through thousands of
indicator/exit/regime combinations to find the best architecture, then tunes per-symbol
parameters.

```bash
python apex.py --concept "momentum breakout with volume" --budget medium
```

## Why This Is Hard (and What I Learned)

Building a strategy optimizer is straightforward. Building one that doesn't lie to you
is an engineering problem with three facets:

**The data integrity problem.** A single misplaced index can leak future information
into a backtest and produce a strategy that works in simulation and collapses live. The
pipeline enforces that signals computed on bar `i` can only trigger fills at the open of
bar `i+1`. Daily and intraday data are fetched independently from Polygon's REST API,
cached locally, and filtered against liquidity floors before optimization.

**The overfitting problem.** Given enough trials, any optimizer will find patterns that
are pure noise. My first version reported a Sharpe of 4.2 on the tune window that
collapsed to 0.3 out-of-sample. The fix was not a smarter optimizer — it was the
discipline of never letting the optimizer see the data it gets judged on. The pipeline
fights overfitting with four layers of defense:

1. **Walk-forward splits** — every Optuna trial runs on a 70/30 IS/OOS split with
   OOS-weighted fitness, and a divergence gate rejects trials where IS and OOS disagree
2. **Monte Carlo simulation** — shuffles the trade sequence 3,000+ times to measure the
   probability of profit under random reordering
3. **Noise and stress injection** — perturbs prices by +/-5%, shifts timing by one bar,
   and swaps regime models to see if the strategy survives
4. **Final holdout** — 25% of the most recent data is reserved before any optimization
   starts. The headline numbers in the report come from this holdout, not the tune window

**The integration problem.** Coordinating a rate-limited API, a long-running Bayesian
optimizer, crash-recovery checkpointing, a statistical validation suite, a
de-correlation filter, and report generation in a single reproducible pipeline is more
engineering than it sounds. A large fraction of the code is the glue: Polygon retry
logic with exponential backoff, config-driven budget profiles, a checkpoint system that
survives crashes, and a clean data-boundary between tune and holdout windows across every
function.

## Pipeline Architecture

```
     +---------------------------+
     | 1. Universe Selection     |   config or --strategy file
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 2. Liquidity Screen       |   daily data, price/volume filter
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 3. Data Fetch +           |   pull execution bars, split off
     |    Holdout Split          |   25% HOLDOUT (untouched)
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 4. LAYER 1                |   Optuna architecture search
     |    (or Strategy Import)   |   (skipped in --strategy mode)
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 5. LAYER 2                |   per-symbol parameter tuning
     |    Walk-Forward Tune      |   with IS/OOS validation
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 6. LAYER 3                |   Monte Carlo + noise injection
     |    Robustness Gauntlet    |   + stress test + sensitivity
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 7. Correlation Filter     |   de-correlate, sector cap
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 8. Final Backtest         |   tune window + TRUE HOLDOUT
     |    (honest numbers)       |   headline comes from holdout
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 9. HTML Report            |   Plotly charts, trade journal
     |    + CSV + JSON + AFL     |   robustness diagnostics
     +---------------------------+
```

## How Trades Are Verified

Every trade produced by the pipeline satisfies these invariants:

- **PnL = (exit - entry) / entry - commission**, computed and cross-checked per trade
- **MFE/MAE** (max favorable / adverse excursion) tracked bar-by-bar from the actual
  High/Low within each position
- **Exit reasons** come directly from the strategy's exit function (Trail Stop, Trend
  Break, etc.) — no translation or approximation
- **Entry scores** are preserved at full decimal precision from the strategy's signal
- **Commission** is deducted at 0.05% per side (configurable)

The report headline always uses the **holdout window** (data the optimizer never saw)
when it has enough trades. A diagnostic table shows tune-vs-holdout side by side so the
reader can see exactly how much of the optimized edge survives on unseen data.

## Tech Stack

| Component | Role |
|-----------|------|
| **Python 3.11+** | Language |
| **[Optuna](https://optuna.org/)** | Bayesian hyperparameter optimization (TPE sampler) |
| **Pandas / NumPy** | Data manipulation and vectorized computation |
| **[Polygon.io](https://polygon.io/) REST API** | Institutional-grade historical market data |
| **[Plotly](https://plotly.com/python/)** | Interactive HTML charts (equity curves, heatmaps) |
| **AmiBroker COM (optional)** | Handoff to live charting environment |

## Quick Start

```bash
# 1. Install
git clone https://github.com/aaasharma870-art/Optuna-Screener.git
cd Optuna-Screener
pip install -r requirements.txt

# 2. Configure — edit apex_config.json with your Polygon API key

# 3. Run with your own strategy
python apex.py --strategy path/to/your_strategy.py --budget light --no-amibroker

# 4. Or let the pipeline discover strategies
python apex.py --concept "mean reversion with volume" --budget medium

# 5. Quick test (3 symbols, ~2 minutes)
python apex.py --test
```

### Adding Tunable Parameters

To enable Optuna parameter tuning for your strategy, add a `PARAMS` dict (default
values your entry/exit functions read from) and a `TUNABLE_PARAMS` dict (search ranges):

```python
PARAMS = {"rs_threshold": 2, "stop_atr_mult": 1.5, "trail_mult": 1.0}

TUNABLE_PARAMS = {
    "rs_threshold": (0.5, 5.0),
    "stop_atr_mult": (0.8, 2.5),
    "trail_mult": (0.5, 2.0),
}

def entry_fn(r, prev, prev2, sym, df, idx):
    p = PARAMS
    if r['RS_21d'] < p['rs_threshold']:
        return None
    ...
```

## Report Output

The HTML report is organized into six tabs:

- **Summary** — headline stats from the holdout window, tune-vs-holdout diagnostic
  table, exit reason breakdown, architecture description
- **Equity & Returns** — interactive equity curve with SPY benchmark overlay, drawdown
  chart, monthly returns heatmap
- **Per-Symbol** — sortable table with tune and holdout columns for every symbol
- **Trade Journal** — last 200 trades with entry/exit prices, PnL, bars held, exit
  reason, and direction
- **Robustness** — Monte Carlo probability of profit, noise retention, stress
  retention, parameter sensitivity, and composite score per symbol
- **Optimization** — parameter stability across symbols

## Limitations and Honest Disclaimers

- **This is per-symbol backtesting, not portfolio simulation.** Each symbol is tested
  independently. The pipeline does not enforce cross-symbol position limits or capital
  allocation — it tells you which symbols your strategy works on, not how a
  multi-position portfolio would perform.
- **Intraday data from Polygon may differ from your broker's fills.** Execution-level
  slippage, partial fills, and order routing are not modeled.
- **The holdout is the most honest number, but it is still a backtest.** No backtest,
  however rigorous, is a guarantee of future performance. Walk-forward validation and
  Monte Carlo testing reduce the risk of overfitting but cannot eliminate it.
- **Low trade counts reduce statistical significance.** Strategies that produce fewer
  than ~30 trades on the holdout window should be interpreted with caution.

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

This is a research and educational tool. It is **not** investment advice, and no result
produced by this pipeline should be interpreted as a guarantee or prediction of future
market performance. Backtests are subject to survivorship, selection, and look-ahead
biases that statistical rigor can only partially mitigate. Use at your own risk.
