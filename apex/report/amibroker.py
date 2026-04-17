"""AmiBroker AFL generation and COM push."""

from datetime import datetime
from pathlib import Path

from apex.logging_util import log


def generate_apex_afl(sorted_syms, results, architecture):
    """
    Generate an AmiBroker AFL formula string with per-symbol optimized
    parameters, indicator computations, regime shading, buy/sell arrows, and
    exploration columns. Pure generic indicator logic - no extra gates.
    """
    per_symbol = results.get("per_symbol", {})
    indicators = architecture.get("indicators", [])

    sym_blocks = []
    for sym in sorted_syms:
        if sym not in per_symbol:
            continue
        p = per_symbol[sym].get("params", {})
        block_lines = [f'    if (Name() == "{sym}")']
        block_lines.append("    {")
        for k, v in sorted(p.items()):
            if isinstance(v, float):
                block_lines.append(f'        {k} = {v:.4f};')
            elif isinstance(v, int):
                block_lines.append(f'        {k} = {v};')
        block_lines.append("    }")
        sym_blocks.append("\n".join(block_lines))

    sym_param_code = "\n    else ".join(sym_blocks)

    ind_code_lines = []
    if "RSI" in indicators:
        ind_code_lines.append("rsi_val = RSI(rsi_period);")
        ind_code_lines.append("rsi_bull = rsi_val < rsi_oversold;")
        ind_code_lines.append("rsi_bear = rsi_val > rsi_overbought;")
    if "MACD" in indicators:
        ind_code_lines.append("macd_line = MACD(macd_fast, macd_slow);")
        ind_code_lines.append("signal_line = Signal(macd_fast, macd_slow, macd_signal);")
        ind_code_lines.append("macd_bull = Cross(macd_line, signal_line);")
        ind_code_lines.append("macd_bear = Cross(signal_line, macd_line);")
    if "Bollinger" in indicators:
        ind_code_lines.append("bb_top = BBandTop(C, boll_period, boll_std);")
        ind_code_lines.append("bb_bot = BBandBot(C, boll_period, boll_std);")
        ind_code_lines.append("boll_bull = C <= bb_bot;")
        ind_code_lines.append("boll_bear = C >= bb_top;")
    if "EMA_Cross" in indicators:
        ind_code_lines.append("ema_f = EMA(C, ema_fast);")
        ind_code_lines.append("ema_s = EMA(C, ema_slow);")
        ind_code_lines.append("ema_bull = Cross(ema_f, ema_s);")
        ind_code_lines.append("ema_bear = Cross(ema_s, ema_f);")
    if "Stochastic" in indicators:
        ind_code_lines.append("stoch_k_val = StochK(stoch_k, stoch_d);")
        ind_code_lines.append("stoch_d_val = StochD(stoch_k, stoch_d);")
        ind_code_lines.append("stoch_bull = stoch_k_val < 20 AND stoch_k_val > stoch_d_val;")
        ind_code_lines.append("stoch_bear = stoch_k_val > 80 AND stoch_k_val < stoch_d_val;")
    if "ADX" in indicators:
        ind_code_lines.append("adx_val = ADX(adx_period);")
        ind_code_lines.append("adx_trending = adx_val > adx_threshold;")
    if "CCI" in indicators:
        ind_code_lines.append("cci_val = CCI(cci_period);")
        ind_code_lines.append("cci_bull = cci_val < cci_oversold;")
        ind_code_lines.append("cci_bear = cci_val > cci_overbought;")
    if "Keltner" in indicators:
        ind_code_lines.append("kelt_mid = EMA(C, keltner_period);")
        ind_code_lines.append("kelt_atr = ATR(14);")
        ind_code_lines.append("kelt_upper = kelt_mid + keltner_mult * kelt_atr;")
        ind_code_lines.append("kelt_lower = kelt_mid - keltner_mult * kelt_atr;")
        ind_code_lines.append("kelt_bull = C <= kelt_lower;")
        ind_code_lines.append("kelt_bear = C >= kelt_upper;")
    if "VolumeSurge" in indicators:
        ind_code_lines.append("vol_ma = MA(V, volume_surge_ma);")
        ind_code_lines.append("vol_surge_sig = V > vol_ma * volume_surge_mult;")
        ind_code_lines.append("vol_bull = vol_surge_sig AND C > O;")
        ind_code_lines.append("vol_bear = vol_surge_sig AND C < O;")
    if "VWAP" in indicators:
        ind_code_lines.append("tp = (H + L + C) / 3;")
        ind_code_lines.append("newday = Day() != Ref(Day(), -1);")
        ind_code_lines.append("cum_tpv = SumSince(newday, tp * V);")
        ind_code_lines.append("cum_v = SumSince(newday, V);")
        ind_code_lines.append("vwap_val = IIf(cum_v > 0, cum_tpv / cum_v, C);")
        ind_code_lines.append("vwap_bull = C > vwap_val;")
        ind_code_lines.append("vwap_bear = C < vwap_val;")
    if "OBV" in indicators:
        ind_code_lines.append("obv_val = OBV();")
        ind_code_lines.append("obv_ma_val = MA(obv_val, obv_ma_period);")
        ind_code_lines.append("obv_bull = obv_val > obv_ma_val;")
        ind_code_lines.append("obv_bear = obv_val < obv_ma_val;")
    if "WilliamsR" in indicators:
        ind_code_lines.append("wr_val = (HHV(H, willr_period) - C) / (HHV(H, willr_period) - LLV(L, willr_period)) * -100;")
        ind_code_lines.append("wr_bull = wr_val < willr_oversold;")
        ind_code_lines.append("wr_bear = wr_val > willr_overbought;")

    ind_code = "\n    ".join(ind_code_lines)

    score_parts = []
    for ind in indicators:
        lname = ind.lower()
        if ind == "EMA_Cross":
            lname = "ema"
        elif ind == "VolumeSurge":
            lname = "vol"
        score_parts.append(f"{lname}_bull")
    score_formula = " + ".join(score_parts) if score_parts else "0"

    afl = f"""// ============================================================
// Optuna Screener Optimized Strategy - Auto-generated AFL
// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// Indicators: {', '.join(indicators)}
// ============================================================

// --- Default Params (overridden per-symbol below) ---
rsi_period = 14; rsi_oversold = 30; rsi_overbought = 70;
macd_fast = 12; macd_slow = 26; macd_signal = 9;
boll_period = 20; boll_std = 2.0;
stoch_k = 14; stoch_d = 3;
obv_ma_period = 20;
adx_period = 14; adx_threshold = 25;
cci_period = 20; cci_oversold = -100; cci_overbought = 100;
willr_period = 14; willr_oversold = -80; willr_overbought = -20;
keltner_period = 20; keltner_mult = 2.0;
volume_surge_ma = 20; volume_surge_mult = 1.5;
ema_fast = 9; ema_slow = 21;
atr_stop_mult = 1.5; atr_target_mult = 2.5;
atr_trail_mult = 1.0; trail_activate_atr = 1.0;
max_hold_bars = 35;
min_score = {architecture.get('min_score', 4)};

// --- Per-Symbol Optimized Params ---
{sym_param_code}

// --- Indicator Computation ---
{ind_code}

// --- Composite Score ---
score = {score_formula};

// --- Regime (simplified: EMA trend) ---
regime_ema20 = EMA(C, 20);
regime_ema50 = EMA(C, 50);
regime_bull = regime_ema20 > regime_ema50;

// --- Entry / Exit ---
entry_atr = ATR(14);
Buy = score >= min_score AND regime_bull;
BuyPrice = Close;

trail_stop = Highest(H, BarsSince(Buy)) - atr_trail_mult * entry_atr;
target_price = ValueWhen(Buy, BuyPrice) + atr_target_mult * ValueWhen(Buy, entry_atr);
stop_price = ValueWhen(Buy, BuyPrice) - atr_stop_mult * ValueWhen(Buy, entry_atr);

Sell = L <= trail_stop OR H >= target_price OR L <= stop_price OR
       BarsSince(Buy) >= max_hold_bars OR NOT regime_bull;
SellPrice = Close;

Buy = ExRem(Buy, Sell);
Sell = ExRem(Sell, Buy);

// --- Charting ---
SetChartOptions(0, chartShowArrows | chartShowDates);
_SECTION_BEGIN("Optuna Screener Price");
Plot(C, "Close", colorDefault, styleCandle);
Plot(regime_ema20, "EMA20", colorYellow, styleLine);
Plot(regime_ema50, "EMA50", colorOrange, styleLine);

clr = IIf(regime_bull, ColorBlend(colorGreen, colorBlack, 0.85),
                        ColorBlend(colorRed, colorBlack, 0.85));
Plot(1, "", clr, styleArea | styleOwnScale | styleNoLabel, 0, 1);

PlotShapes(IIf(Buy, shapeUpArrow, shapeNone), colorBrightGreen, 0, L, -20);
PlotShapes(IIf(Sell, shapeDownArrow, shapeNone), colorRed, 0, H, -20);
_SECTION_END();

// --- Exploration Columns ---
Filter = Buy OR Sell;
AddTextColumn(WriteIf(Buy, "BUY", "SELL"), "Signal");
AddColumn(C, "Price", 1.2);
AddColumn(score, "Score", 1.0);
AddColumn(entry_atr, "ATR", 1.4);
AddColumn(target_price, "Target", 1.2);
AddColumn(stop_price, "Stop", 1.2);
AddTextColumn(WriteIf(regime_bull, "BULLISH", "BEARISH"), "Regime");
"""

    return afl


def push_to_amibroker(results, afl_str, output_dir, cfg):
    """
    Push results to AmiBroker via COM automation. Falls back gracefully to
    file-based output if pywin32 or AmiBroker are not available.
    """
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    sorted_syms = results.get("sorted_syms", [])

    afl_path = od / "OptunaScreener_Strategy.afl"
    with open(afl_path, "w", encoding="utf-8") as f:
        f.write(afl_str)
    log(f"AFL saved: {afl_path}")

    tls_path = od / "OptunaScreener_Watchlist.tls"
    with open(tls_path, "w") as f:
        for sym in sorted_syms:
            f.write(f"{sym}\n")
    log(f"Watchlist saved: {tls_path}")

    com_success = False
    try:
        import win32com.client
        ab = win32com.client.Dispatch("Broker.Application")
        log("Connected to AmiBroker via COM")

        ab_path = None
        try:
            ab_path = ab.DatabasePath
        except Exception:
            pass
        if ab_path:
            formulas_dir = Path(ab_path).parent / "Formulas" / "Custom"
            formulas_dir.mkdir(parents=True, exist_ok=True)
            dest_afl = formulas_dir / "OptunaScreener_Strategy.afl"
            with open(dest_afl, "w", encoding="utf-8") as f:
                f.write(afl_str)
            log(f"AFL copied to AmiBroker: {dest_afl}")

        try:
            wl_idx = ab.AddWatchList("OptunaScreener_Picks")
            for sym in sorted_syms:
                ab.AddToWatchList(wl_idx, sym)
            log(f"Watchlist 'OptunaScreener_Picks' created with {len(sorted_syms)} symbols")
        except Exception as e:
            log(f"Watchlist creation via COM failed: {e}", "WARN")

        ab.RefreshAll()
        com_success = True
        log("AmiBroker COM push complete")

    except ImportError:
        log("win32com not available - COM push skipped. AFL saved to disk.", "WARN")
    except Exception as e:
        log(f"AmiBroker COM error: {e}. AFL saved to disk.", "WARN")

    return com_success
