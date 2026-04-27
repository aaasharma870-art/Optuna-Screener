"""Strategies that have NOT passed true-OOS validation.

Two categories live here:

1. NO-REAL-DATA-OOS — strategies with only synthetic unit tests:
     - ts_exhaustion_fade
     - vix_liquidity_reclaim
     - institutional_arbitrage_engine_v2
     - advanced_compounder_v11

2. FAILED — strategies that ran on real data and either lost money or could
   not produce signals without the ~200hr options-chain ingest:
     - smc_structural        (lost money tune AND holdout)
     - vrp_gex_fade          (needs options chain)
     - opex_gravity          (needs options chain)
     - vol_skew_arb          (needs options chain)

These manifests are documentation only. Nothing in apex_config.json's default
ensemble loads them. Do NOT promote any of them to strategies/tested/ until a
Phase-15-style 75/25 holdout on real SPY/QQQ data shows Sharpe > 1 OOS.
"""
