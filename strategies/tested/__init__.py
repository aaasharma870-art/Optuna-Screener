"""Strategies that pass true-OOS validation (75/25 holdout, CPCV-aware).

Membership rule: a strategy lives here only after a Phase-15-style honest
holdout run on real SPY/QQQ data shows Sharpe > 1 OOS. Synthetic-data unit
tests do not qualify. See strategies/untested/ for everything else.
"""
