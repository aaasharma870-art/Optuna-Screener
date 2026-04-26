"""Ensemble HTML report generator (Phase 12I).

Self-contained HTML with 7 tabs: Headline, Per-Strategy Contributions,
Equity Curves, Regime Breakdown, CPCV Distribution, Walk-Forward Weights,
Layer A Results.

Uses Plotly (CDN) embedded into a single HTML file for easy distribution.
"""
import html
import json
from pathlib import Path
from typing import Any, Dict, List


_TAB_LABELS = [
    "Headline",
    "Per-Strategy Contributions",
    "Equity Curves",
    "Regime Breakdown",
    "CPCV Distribution",
    "Walk-Forward Weights",
    "Layer A Results",
]


def _esc(x) -> str:
    return html.escape(str(x))


def _equity_from_returns(returns: List[float], start: float = 100.0) -> List[float]:
    out = [start]
    cur = start
    for r in returns:
        try:
            cur = cur * (1.0 + float(r))
        except Exception:
            pass
        out.append(cur)
    return out


def _equity_from_positions(positions: List[float],
                           closes: List[float],
                           start: float = 100.0) -> List[float]:
    """Reconstruct an equity curve from per-bar positions and a price series."""
    if not positions or not closes:
        return [start]
    n = min(len(positions), len(closes))
    eq = [start]
    cur = start
    for i in range(1, n):
        try:
            r = (closes[i] - closes[i - 1]) / closes[i - 1]
        except Exception:
            r = 0.0
        # signal at i-1 fills at i
        cur = cur * (1.0 + positions[i - 1] * r)
        eq.append(cur)
    return eq


def generate_ensemble_report(results: Dict[str, Any], output_dir: str) -> str:
    """Render an ensemble-mode HTML report.

    Required keys in `results`:
      primary_symbol, weights, strategies, layer_a_rows, layer_a_by_strategy,
      layer_b, layer_c, ref_close, ref_dt, portfolio_position,
      per_strategy_positions, current_regime, run_info.
    """
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    out_path = od / "ensemble_report.html"

    primary = results.get("primary_symbol", "")
    weights: Dict[str, float] = results.get("weights", {})
    strategies: List[str] = results.get("strategies", [])
    layer_a_rows: List[Dict[str, Any]] = results.get("layer_a_rows", [])
    layer_a_by_strat: Dict[str, str] = results.get("layer_a_by_strategy", {})
    layer_b: Dict[str, Any] = results.get("layer_b", {}) or {}
    layer_c: Dict[str, Any] = results.get("layer_c", {}) or {}
    ref_close = results.get("ref_close", []) or []
    ref_dt = results.get("ref_dt", []) or []
    portfolio_position = results.get("portfolio_position", []) or []
    per_strategy_positions = results.get("per_strategy_positions", {}) or {}
    current_regime = results.get("current_regime", "UNKNOWN")
    run_info = results.get("run_info", {}) or {}

    # ---- Headline numbers ----
    sharpe_med = layer_b.get("sharpe_median", 0.0) or 0.0
    sharpe_iqr = layer_b.get("sharpe_iqr", [0.0, 0.0]) or [0.0, 0.0]
    pct_pos = layer_b.get("sharpe_pct_positive", 0.0) or 0.0
    layer_b_status = layer_b.get("layer_b_status", "?")
    layer_c_status = layer_c.get("layer_c_status", "?")
    portfolio_returns = layer_b.get("portfolio_returns", []) or []
    portfolio_eq = _equity_from_returns(portfolio_returns)
    if portfolio_eq:
        peak = portfolio_eq[0]
        max_dd = 0.0
        for v in portfolio_eq:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        total_return_pct = (portfolio_eq[-1] / portfolio_eq[0] - 1.0) * 100.0
    else:
        max_dd = 0.0
        total_return_pct = 0.0

    # Layer A: collapse per-(strategy,symbol) to per-strategy summary
    layer_a_summary: Dict[str, Dict[str, Any]] = {}
    for r in layer_a_rows:
        n = r["strategy_name"]
        ent = layer_a_summary.setdefault(n, {
            "median_sharpe_sum": 0.0, "n": 0, "pass": 0, "fail": 0,
        })
        ent["median_sharpe_sum"] += float(r.get("median_sharpe", 0.0))
        ent["n"] += 1
        if r.get("layer_a_status") == "PASS":
            ent["pass"] += 1
        else:
            ent["fail"] += 1
    layer_a_status_overall = (
        "PASS" if all(v.get("pass", 0) >= v.get("fail", 0)
                       for v in layer_a_summary.values()) else "MIXED"
    )

    # ---- Tab 1: Headline ----
    headline_html = f"""
      <div class="card-grid">
        <div class="card"><div class="lbl">Primary Symbol</div><div class="val">{_esc(primary)}</div></div>
        <div class="card"><div class="lbl">Current Regime</div><div class="val">{_esc(current_regime)}</div></div>
        <div class="card"><div class="lbl">Median CPCV Sharpe (Layer B)</div><div class="val">{sharpe_med:.2f}</div></div>
        <div class="card"><div class="lbl">CPCV Sharpe IQR</div><div class="val">[{sharpe_iqr[0]:.2f}, {sharpe_iqr[1]:.2f}]</div></div>
        <div class="card"><div class="lbl">% Positive Folds</div><div class="val">{pct_pos*100:.0f}%</div></div>
        <div class="card"><div class="lbl">Total Return (sample)</div><div class="val">{total_return_pct:+.2f}%</div></div>
        <div class="card"><div class="lbl">Max Drawdown</div><div class="val">{-max_dd*100:.2f}%</div></div>
        <div class="card status-{_esc(layer_a_status_overall.lower())}"><div class="lbl">Layer A</div><div class="val">{_esc(layer_a_status_overall)}</div></div>
        <div class="card status-{_esc(layer_b_status.lower())}"><div class="lbl">Layer B</div><div class="val">{_esc(layer_b_status)}</div></div>
        <div class="card status-{_esc(layer_c_status.lower())}"><div class="lbl">Layer C</div><div class="val">{_esc(layer_c_status)}</div></div>
      </div>
    """

    # ---- Tab 2: Per-Strategy Contributions ----
    contrib_rows = ""
    for name in strategies:
        w = weights.get(name, 0.0)
        a_med = 0.0
        s = layer_a_summary.get(name)
        if s and s["n"]:
            a_med = s["median_sharpe_sum"] / s["n"]
        # Approximate return contribution: weight * mean(per-bar position * px return)
        contrib_pct = 0.0
        pos = per_strategy_positions.get(name, [])
        if pos and ref_close:
            n_ = min(len(pos), len(ref_close))
            cum = 1.0
            for i in range(1, n_):
                try:
                    r_ = (ref_close[i] - ref_close[i - 1]) / ref_close[i - 1]
                except Exception:
                    r_ = 0.0
                cum *= (1.0 + pos[i - 1] * r_)
            contrib_pct = (cum - 1.0) * 100.0
        weighted_contrib = contrib_pct * w
        contrib_rows += (
            f"<tr><td>{_esc(name)}</td>"
            f"<td>{w:.3f}</td>"
            f"<td>{a_med:+.2f}</td>"
            f"<td>{contrib_pct:+.2f}%</td>"
            f"<td>{weighted_contrib:+.2f}%</td></tr>\n"
        )
    contrib_html = f"""
      <table class="tbl">
        <thead><tr><th>Strategy</th><th>Weight</th><th>Layer A Median Sharpe</th>
        <th>Solo Return</th><th>Weighted Return</th></tr></thead>
        <tbody>
          {contrib_rows}
        </tbody>
      </table>
    """

    # ---- Tab 3: Equity Curves ----
    portfolio_curve = _equity_from_returns(portfolio_returns)
    portfolio_x = list(range(len(portfolio_curve)))

    eq_traces = [{
        "x": portfolio_x,
        "y": [round(v, 4) for v in portfolio_curve],
        "type": "scatter", "mode": "lines",
        "name": "Combined Portfolio",
        "line": {"color": "#2ea44f", "width": 3},
    }]
    for name, pos in per_strategy_positions.items():
        if name == "cross_asset_vol_overlay":
            continue
        eq = _equity_from_positions(list(pos), list(ref_close))
        eq_traces.append({
            "x": list(range(len(eq))),
            "y": [round(v, 4) for v in eq],
            "type": "scatter", "mode": "lines",
            "name": name,
            "line": {"width": 1.2},
        })
    equity_div = "ensemble_equity_div"
    equity_traces_json = json.dumps(eq_traces)

    # ---- Tab 4: Regime Breakdown ----
    regime_returns: Dict[str, float] = {"R1": 0.0, "R2": 0.0, "R3": 0.0, "R4": 0.0}
    for i, r in enumerate(portfolio_returns):
        # Without per-bar regime here, attribute everything to the dominant regime
        regime_returns[current_regime if current_regime in regime_returns else "R2"] += r
    regime_div = "ensemble_regime_div"
    regime_traces = [{
        "x": list(regime_returns.keys()),
        "y": [round(v * 100, 4) for v in regime_returns.values()],
        "type": "bar",
        "marker": {"color": ["#2ea44f", "#58a6ff", "#f0b400", "#d73a49"]},
        "name": "Return %",
    }]
    regime_traces_json = json.dumps(regime_traces)

    # ---- Tab 5: CPCV Distribution ----
    cpcv_sharpes = layer_b.get("oos_sharpes", []) or []
    cpcv_div = "ensemble_cpcv_div"
    cpcv_traces = [{
        "x": [round(s, 3) for s in cpcv_sharpes],
        "type": "histogram",
        "marker": {"color": "#58a6ff"},
        "name": "OOS Sharpe per Fold",
    }]
    cpcv_traces_json = json.dumps(cpcv_traces)

    # ---- Tab 6: Walk-Forward Weights ----
    static_sharpe = layer_c.get("static_sharpe", 0.0) or 0.0
    dynamic_sharpe = layer_c.get("dynamic_sharpe", 0.0) or 0.0
    uplift = layer_c.get("uplift", 0.0) or 0.0
    n_months = layer_c.get("n_months", 0) or 0
    wf_div = "ensemble_wf_div"
    wf_traces = [
        {"x": ["static", "dynamic"],
         "y": [round(static_sharpe, 3), round(dynamic_sharpe, 3)],
         "type": "bar",
         "marker": {"color": ["#6e7681", "#2ea44f"]},
         "name": "Sharpe (annualized)"},
    ]
    wf_traces_json = json.dumps(wf_traces)
    wf_summary_html = f"""
      <p>n_months = {n_months} | static = {static_sharpe:.2f} |
         dynamic = {dynamic_sharpe:.2f} | uplift = {uplift:+.2f}</p>
    """

    # ---- Tab 7: Layer A Results ----
    la_rows = ""
    for r in layer_a_rows:
        st = r.get("layer_a_status", "?")
        la_rows += (
            f"<tr><td>{_esc(r.get('strategy_name',''))}</td>"
            f"<td>{_esc(r.get('symbol',''))}</td>"
            f"<td>{r.get('n_folds',0)}</td>"
            f"<td>{r.get('median_sharpe',0.0):+.2f}</td>"
            f"<td>[{r.get('iqr_low',0.0):+.2f}, {r.get('iqr_high',0.0):+.2f}]</td>"
            f"<td>{r.get('pct_positive',0.0)*100:.0f}%</td>"
            f"<td class='status-{_esc(st.lower())}'>{_esc(st)}</td></tr>\n"
        )
    la_html = f"""
      <table class="tbl">
        <thead><tr><th>Strategy</th><th>Symbol</th><th>Folds</th>
          <th>Median Sharpe</th><th>IQR</th><th>% Positive</th><th>Status</th></tr></thead>
        <tbody>
          {la_rows}
        </tbody>
      </table>
    """

    # ---- Build full HTML ----
    tabs_buttons = "\n".join(
        f'<button class="tab-btn" data-tab="tab-{i}">{_esc(lbl)}</button>'
        for i, lbl in enumerate(_TAB_LABELS)
    )

    timestamp = run_info.get("timestamp", "")
    concept = run_info.get("concept", "Multi-strategy ensemble")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Optuna Screener - Ensemble Report</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
           background: #0d1117; color: #c9d1d9; margin: 0; padding: 24px; }}
    h1 {{ color: #58a6ff; margin: 0 0 4px 0; }}
    .sub {{ color: #8b949e; margin-bottom: 16px; }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 1px solid #30363d; }}
    .tab-btn {{ background: #161b22; border: 1px solid #30363d; color: #c9d1d9;
               padding: 8px 14px; cursor: pointer; border-radius: 4px 4px 0 0; }}
    .tab-btn.active {{ background: #1f6feb; color: #fff; border-color: #1f6feb; }}
    .tab-pane {{ display: none; padding: 16px; background: #161b22; border: 1px solid #30363d;
                border-top: none; }}
    .tab-pane.active {{ display: block; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                 gap: 12px; }}
    .card {{ background: #0d1117; border: 1px solid #30363d; padding: 12px; border-radius: 6px; }}
    .card .lbl {{ color: #8b949e; font-size: 12px; }}
    .card .val {{ color: #c9d1d9; font-size: 22px; font-weight: 600; margin-top: 4px; }}
    .card.status-pass {{ border-color: #2ea44f; }}
    .card.status-pass .val {{ color: #2ea44f; }}
    .card.status-fail {{ border-color: #d73a49; }}
    .card.status-fail .val {{ color: #d73a49; }}
    .card.status-mixed {{ border-color: #f0b400; }}
    .card.status-mixed .val {{ color: #f0b400; }}
    .card.status-? {{ border-color: #6e7681; }}
    .tbl {{ border-collapse: collapse; width: 100%; }}
    .tbl th, .tbl td {{ border: 1px solid #30363d; padding: 6px 10px; text-align: left; }}
    .tbl th {{ background: #161b22; color: #58a6ff; }}
    .status-pass {{ color: #2ea44f; }}
    .status-fail {{ color: #d73a49; }}
    .status-error {{ color: #d73a49; }}
    .status-mixed {{ color: #f0b400; }}
  </style>
</head>
<body>
  <h1>Optuna Screener - Ensemble Report</h1>
  <div class="sub">{_esc(concept)} | {_esc(timestamp)} | primary={_esc(primary)}</div>

  <div class="tabs">
    {tabs_buttons}
  </div>

  <div class="tab-pane" id="tab-0">{headline_html}</div>
  <div class="tab-pane" id="tab-1">{contrib_html}</div>
  <div class="tab-pane" id="tab-2"><div id="{equity_div}" style="height:520px;"></div></div>
  <div class="tab-pane" id="tab-3"><div id="{regime_div}" style="height:480px;"></div></div>
  <div class="tab-pane" id="tab-4"><div id="{cpcv_div}" style="height:480px;"></div></div>
  <div class="tab-pane" id="tab-5">{wf_summary_html}<div id="{wf_div}" style="height:420px;"></div></div>
  <div class="tab-pane" id="tab-6">{la_html}</div>

  <script>
    // Tab switching
    var btns = document.querySelectorAll(".tab-btn");
    var panes = document.querySelectorAll(".tab-pane");
    function showTab(i) {{
      btns.forEach(function(b){{ b.classList.remove("active"); }});
      panes.forEach(function(p){{ p.classList.remove("active"); }});
      btns[i].classList.add("active");
      panes[i].classList.add("active");
    }}
    btns.forEach(function(b, i){{ b.addEventListener("click", function(){{ showTab(i); }}); }});
    showTab(0);

    // Render Plotly charts
    var darkLayout = {{ paper_bgcolor: "#161b22", plot_bgcolor: "#0d1117",
                       font: {{ color: "#c9d1d9" }}, margin: {{ t: 36 }}}};
    Plotly.newPlot("{equity_div}", {equity_traces_json},
                   Object.assign({{title: "Equity Curves (combined vs per-strategy)"}}, darkLayout));
    Plotly.newPlot("{regime_div}", {regime_traces_json},
                   Object.assign({{title: "Returns by Regime"}}, darkLayout));
    Plotly.newPlot("{cpcv_div}", {cpcv_traces_json},
                   Object.assign({{title: "Layer B CPCV OOS Sharpe Distribution"}}, darkLayout));
    Plotly.newPlot("{wf_div}", {wf_traces_json},
                   Object.assign({{title: "Layer C Walk-Forward Weights: Static vs Dynamic"}}, darkLayout));
  </script>
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    return str(out_path)
