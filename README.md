# Optuna Screener — A Self-Built Trading Strategy Research Pipeline

> An end-to-end Python pipeline that automatically discovers, optimizes, and validates
> systematic trading strategies using real market data, hyperparameter search, and
> Monte Carlo robustness testing.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Built with Optuna](https://img.shields.io/badge/built%20with-Optuna-8A2BE2)
![Data: Polygon.io](https://img.shields.io/badge/data-Polygon.io-orange)

---

## What This Project Is

This is a complete research pipeline that you run with a single command. It pulls real
historical market data from **Polygon.io** (a professional-grade data provider used by
hedge funds and brokers), then uses **Optuna** — a state-of-the-art hyperparameter
optimization library originally developed by Preferred Networks — to search through
thousands of possible trading-strategy configurations. Every candidate strategy is
validated against unseen data using walk-forward analysis, stress-tested with Monte
Carlo simulations, and finally measured on a held-out window that the optimizer has
never touched. At the end, the pipeline produces a self-contained HTML report with
equity curves, trade journals, robustness diagnostics, and an AmiBroker AFL file ready
to drop into a live charting package.

The whole system — from data fetch to final report — runs from one command:

```bash
python apex.py --concept "trend following with volume confirmation" --budget medium
```

## Why This Is Hard

Building a trading-strategy search pipeline sounds simple until you try to make it
honest. Three problems dominate the engineering work, and each one is addressed
explicitly in this codebase.

**The data problem.** Financial data is noisy, gappy, and easy to misuse. Quotes come
with timestamps that are cheaper than they are correct; a single misplaced index can
leak information from the future into a backtest and produce a "strategy" that works
in simulation and collapses live. The pipeline enforces that signals are always
computed from data with timestamp ≤ bar `i` and that trades fill at the **open of bar
`i+1`** — there is no way for the backtester to see a bar's close before deciding to
trade it. Daily and intraday data are fetched independently from Polygon's REST API,
cached locally, and survivorship-checked against a liquidity floor before any
optimization touches them.

**The overfitting problem.** Given enough trials, any hyperparameter optimizer will
find spurious patterns — that is the whole point of Rob Pardo's warning that
"optimization is the art of self-delusion." This pipeline fights it with a four-layer
defense: (1) a 70/30 walk-forward in-sample / out-of-sample split inside every Optuna
trial, (2) a divergence gate that throws out any trial whose IS and OOS performance
don't agree, (3) a Monte Carlo robustness test that shuffles the trade sequence 3,000
times and requires a high probability of profit under reshuffling, and (4) an
untouched 25% **final holdout window** that is split off *before* optimization starts
and is only looked at once — to report honest numbers. The headline PF and return
shown in the HTML report come from this holdout, not from the tune window.

**The integration problem.** A research pipeline is only useful if it's reproducible
and automatable. That means coordinating a rate-limited third-party API, a long-running
Bayesian optimizer, checkpointing to survive crashes, a statistical validation suite, a
correlation filter that prevents one sector from dominating the portfolio, and report
generation — all in one process with consistent state. A large fraction of the code is
the glue that makes this integration actually work: the Polygon retry/back-off logic,
the config-driven budget profiles, the checkpoint system that lets you resume a
multi-hour run, and the clean separation between tune-window data and holdout data
across every function in the data pipeline.

## Pipeline Architecture

Running `python apex.py` executes the following stages in sequence. Each stage writes
a checkpoint so a crashed run can be resumed with `--resume`.

```
     +---------------------------+
     | 1. Universe Selection     |   config → candidate symbols
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 2. Quick Screen           |   daily data, liquidity filters
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 3. Data Fetch +           |   pull exec bars, split off
     |    Holdout Split          |   25% HOLDOUT (untouched)
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 4. LAYER 1                |   Optuna searches architecture
     |    Architecture Search    |   space (indicators / exits / regime)
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 5. LAYER 2                |   per-symbol deep TPE tune,
     |    Deep Parameter Tune    |   walk-forward IS/OOS split
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 6. LAYER 3                |   Monte Carlo + noise + regime
     |    Robustness Gauntlet    |   stress + param sensitivity
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 7. Correlation Filter     |   de-correlate, sector cap
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 8. Final Backtest         |   run on FULL tune universe
     |    (Tune + HOLDOUT)       |   + on the 25% HOLDOUT window
     +-----------+---------------+
                 |
     +-----------v---------------+
     | 9. Report Generation      |   HTML (Plotly) + CSV + JSON
     |                           |   + AmiBroker .afl
     +---------------------------+
```

**Step-by-step:**

1. **Universe selection** — the config lists target symbols and a "forced" set (e.g.
   `SPY`, `QQQ`) that always enters the universe. Sector labels come from a built-in
   map used later by the correlation filter.

2. **Quick screen** — daily OHLCV is pulled from Polygon and each symbol is filtered
   by price bounds, average volume, and minimum history. Failing symbols are dropped
   before any expensive work happens.

3. **Data fetch + holdout split** — execution-timeframe bars (default: 1-hour) are
   fetched. **The final 25% of bars is immediately split off as a holdout window and
   stored separately.** Nothing from steps 4-7 ever sees it.

4. **Layer 1 — Architecture search** — an Optuna TPE study searches over discrete
   architectural choices: which indicators to combine (from a library of 12), which
   exit methods to use, which regime model to run, and how to aggregate the signals.
   Each trial runs a quick inner optimization on a subset of symbols and reports
   fitness back to the outer study.

5. **Layer 2 — Deep parameter tuning** — for every symbol that survived the screen,
   a per-symbol Optuna study tunes the numerical parameters of the best architecture.
   Each trial does a 70/30 walk-forward split and blends in-sample and out-of-sample
   fitness (OOS-weighted), rejecting trials whose IS/OOS divergence exceeds 80%.

6. **Layer 3 — Robustness gauntlet** — every surviving strategy is stress-tested:
   - **Monte Carlo** — shuffle the trade sequence 3,000+ times and report the
     probability of net profit, the 5th/50th/95th-percentile final equity, and the
     95th-percentile drawdown.
   - **Noise injection** — jitter closes by ±5% and shift by one bar; strategies whose
     PF collapses under noise are rejected.
   - **Regime stress** — swap the regime classifier for a different one and see
     whether the strategy survives.
   - **Parameter sensitivity** — perturb every numerical parameter by ±10% and
     measure PF range; rewards strategies that live on a plateau, not a knife-edge.

7. **Correlation filter** — pairwise trade-return correlations are computed, and for
   any pair with |correlation| > 0.70 the lower-fitness symbol is rejected. A sector
   cap (default: 3 per sector) prevents concentration.

8. **Final backtest** — re-runs the strategy on the FULL tune window (for the
   optimized universe, not just the correlation-filter survivors — so headline stats
   aren't inflated by post-selection bias) and **on the 25% final holdout window**
   that the optimizer never touched. The holdout numbers are the headline.

9. **Report generation** — a single self-contained `report.html` with Plotly charts,
   plus `trades.csv`, `summary.csv`, and `parameters.json`. An AmiBroker `.afl` file
   with per-symbol optimized parameters is emitted for handoff to a live charting
   environment (pushed via COM on Windows if AmiBroker is installed).

## Statistical Rigor — Why You Should Trust the Numbers

Most amateur backtests fail the same way: they report a number from the window the
strategy was tuned on, and then that number evaporates on live data. This pipeline is
structured so that the headline number in the HTML report is the one the optimizer
**could not cheat on**. Every defense below is implemented in code and is always-on.

- **Look-ahead bias prevention.** Every signal is computed from data with
  timestamp ≤ bar `i` and every fill happens at `open[i+1]`. You cannot see a bar's
  close before trading it.

- **Walk-forward optimization.** Inside every Layer-2 Optuna trial, the tune window
  is split 70% in-sample / 30% out-of-sample. The fitness reported to Optuna is a
  blend that weights OOS more heavily than IS. A divergence gate rejects trials
  whose IS and OOS fitness don't agree — the signature of memorization.

- **Final holdout window.** An additional **25% of the most recent data** is
  reserved *before any optimization phase runs*. Layers 1, 2, and 3 only see the
  remaining 75%. After optimization is done, the pipeline runs one more backtest on
  the holdout and reports those numbers as the headline. This is the single most
  important feature in the whole codebase.

- **Monte Carlo robustness.** Each surviving strategy's trade pnls are shuffled
  3,000 times. The pipeline records the probability that a net-profitable equity
  curve emerges, plus the 95th-percentile drawdown. Low prob-of-profit = reject.

- **Selection-bias correction.** When the final report computes portfolio stats, it
  uses the **full optimized universe**, not just the symbols that passed the
  correlation filter. The survivor subset is reported separately and clearly
  labeled as post-selection-biased. This prevents the common report-time inflation
  where "I optimized 20 symbols, kept the 5 that worked, and called that my
  strategy's performance."

- **Per-symbol overfitting controls.** Trial budgets are capped by a config-driven
  `--budget` flag (`light` / `medium` / `heavy`), an upper-bound PF gate rejects
  implausibly perfect curves, and a minimum-trade floor rejects candidates that only
  produced a handful of lucky trades.

- **Tune-vs-holdout decay table.** The HTML report always shows a diagnostic table
  comparing the TUNE window to the HOLDOUT window side-by-side, with a "% decay"
  column so the reader can see exactly how much of the optimized edge survives.

## Tech Stack

| Component | Role |
|-----------|------|
| **Python 3.11+** | Language |
| **[Optuna](https://optuna.org/)** | Hyperparameter optimization (TPE sampler, multivariate, pruning) |
| **Pandas / NumPy** | Data manipulation and vectorized math |
| **[Polygon.io](https://polygon.io/) REST API** | Institutional-grade historical and real-time market data |
| **[Plotly](https://plotly.com/python/)** | Self-contained interactive HTML charts |
| **AmiBroker COM via pywin32** | Optional handoff to a live charting environment |
| **Requests** | HTTP client with retry/back-off for the Polygon API |

## How To Run

**1. Get a Polygon.io API key.** Free tier is sufficient for daily data and light
intraday testing. [polygon.io](https://polygon.io/).

**2. Install dependencies.**

```bash
pip install -r requirements.txt
```

**3. Edit `apex_config.json`.** Paste your Polygon API key into `polygon_api_key`,
then edit the `target_symbols` list to choose your research universe. The rest of
the config can stay at defaults.

**4. Run the pipeline.**

```bash
# Quick test run (3 symbols, tiny budget, ~2 minutes)
python apex.py --test

# Full medium run
python apex.py --concept "trend following momentum breakout" --budget medium

# Resume from a checkpoint after a crash
python apex.py --resume

# Skip the AmiBroker handoff if you don't use AmiBroker
python apex.py --no-amibroker
```

**5. Open the report.** At the end of the run the HTML report auto-opens in your
default browser. The output directory also contains `trades.csv`, `summary.csv`,
`parameters.json`, and an `OptunaScreener_Strategy.afl` file.

## Example Output

The HTML report is laid out as a tabbed dashboard:

- **Executive Summary** — headline TRUE HOLDOUT stats (total return, PF, win rate,
  trades, max drawdown, Sharpe, Sortino, edge ratio, avg win/loss), plus the
  architecture the optimizer settled on (indicator set, exit methods, regime model,
  aggregation mode, minimum entry score, execution timeframe).
- **Equity & Returns** — interactive Plotly equity curve with a SPY benchmark
  overlay, a drawdown-underwater plot, and a monthly-returns heatmap.
- **Per-Symbol** — sortable table of trades / PF / win rate / return / DD / Sharpe /
  robustness composite for every optimized symbol.
- **Trade Journal** — the last 200 trades with entry/exit timestamps, prices, PnL,
  bars held, exit reason, and entry regime.
- **Optimization** — parameter stability table showing which hyperparameters stayed
  within a tight PF band under ±10% perturbation (high stability = less likely to
  be overfit).
- **Robustness** — per-symbol breakdown of Monte Carlo prob-of-profit,
  noise-injection PF retention, regime-stress PF retention, and parameter
  sensitivity, combined into a composite robustness score.

Above the tabs, a **TUNE vs HOLDOUT diagnostic table** shows the exact performance
decay from the tune window to the unseen holdout — the most important number in the
whole report.

## What I Learned Building This

- **Why naive Optuna optimization is dangerous.** My first version of this pipeline
  reported a Sharpe of 4.2 on the tune window and collapsed to 0.3 out-of-sample.
  That's what motivated the four-layer defense (walk-forward IS/OOS + divergence
  gate + Monte Carlo + final holdout). The fix wasn't a smarter optimizer — it was
  the discipline of never letting the optimizer see the data it was being judged
  on.

- **How to design API rate-limited workflows.** Polygon's rate limits punish naive
  callers with 429s. I built a retry/back-off layer with exponential wait times and
  aggressive on-disk caching so that a 47-minute optimization run doesn't make the
  same API call twice. Cache hits matter more than CPU.

- **The discipline of separating "tune window" from "true holdout."** The single
  most valuable line in the whole codebase is the `cut = int(len(exec_df_full) *
  (1.0 - holdout_pct))` split in `phase3_fetch_data` — and then the fact that
  every downstream function accepts `(exec_df, exec_df_holdout)` as separate
  arguments so the holdout can never accidentally leak back into training. Holding
  that boundary everywhere in the data pipeline is more engineering than most
  people realize.

- **Why a 47-minute optimization run beats a 5-second curve fit.** Optuna with TPE
  samples the parameter space intelligently, prunes bad trials early, and blends
  IS/OOS fitness — which means even if a single configuration happens to look great
  on the training window, the optimizer is steered away from it because the OOS
  number doesn't agree. Taking longer is the whole point.

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

This is a research and educational tool. It is **not** investment advice, and no
result produced by this pipeline should be interpreted as a guarantee or prediction
of future market performance. Backtests are subject to survivorship, selection, and
look-ahead biases that statistical rigor can only partially mitigate. Use at your
own risk.
