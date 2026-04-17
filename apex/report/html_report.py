"""HTML report generation with embedded Plotly charts."""

import json
from datetime import datetime
from pathlib import Path

from apex.logging_util import log
from apex.util.sector_map import SECTOR_MAP


def generate_html_report(results, architecture, robustness_data, run_info, output_dir):
    """
    Generate a complete self-contained HTML report with embedded Plotly charts.

    The headline number is the TRUE HOLDOUT result (the window the optimizer
    never saw) whenever holdout data has enough trades, with a TUNE-vs-HOLDOUT
    diagnostic table so the reader can see performance decay.
    """
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    holdout_stats = results.get("holdout_universe_stats", {}) or {}
    tune_stats = results.get("portfolio_stats", {}) or {}
    survivor_stats = results.get("survivor_stats", {}) or {}
    holdout_survivor_stats = results.get("holdout_survivor_stats", {}) or {}

    # Only use holdout as headline if it has meaningful data
    use_holdout = holdout_stats.get("trades", 0) >= 5
    headline_stats = holdout_stats if use_holdout else tune_stats
    headline_label = "TRUE OUT-OF-SAMPLE HOLDOUT" if use_holdout else "TUNE WINDOW"

    all_trades = results.get("all_trades", [])
    holdout_trades = results.get("holdout_all_trades", [])
    per_symbol = results.get("per_symbol", {})
    sorted_syms = results.get("sorted_syms", [])
    survivor_syms = results.get("survivor_syms", [])

    # --- Exit reason breakdown (use actual keys from trades) ---
    exit_counts = {}
    for t in all_trades:
        r = t.get("exit_reason", "unknown")
        exit_counts[r] = exit_counts.get(r, 0) + 1
    exit_total = max(1, sum(exit_counts.values()))
    exit_items = ""
    for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
        pct = count / exit_total * 100
        exit_items += f'<div class="exit-bar"><span class="exit-label">{reason}</span><div class="exit-fill" style="width:{pct:.0f}%"></div><span class="exit-pct">{count} ({pct:.1f}%)</span></div>\n'

    # --- Equity curve data ---
    equity_dates = results.get("equity_dates", [])
    equity_values = results.get("equity_values", [])
    eq_dates_json = json.dumps([str(d) for d in equity_dates])
    eq_vals_json = json.dumps([round(v, 2) for v in equity_values])

    benchmark = results.get("benchmark")
    bench_trace = ""
    if benchmark:
        bench_dates = json.dumps([str(d) for d in benchmark["dates"]])
        bench_vals = json.dumps([round(v, 2) for v in benchmark["equity"]])
        bench_trace = f"""{{
            x: {bench_dates}, y: {bench_vals},
            type: 'scatter', mode: 'lines', name: 'SPY Benchmark',
            line: {{color: '#6e7681', dash: 'dot', width: 1}}
        }},"""

    # Drawdown series
    dd_values = []
    peak_eq = 10000.0
    for v in equity_values:
        if v > peak_eq:
            peak_eq = v
        dd = (peak_eq - v) / peak_eq * 100.0
        dd_values.append(round(-dd, 2))
    dd_json = json.dumps(dd_values)

    # --- Monthly returns heatmap ---
    monthly_data = {}
    for t in all_trades + holdout_trades:
        try:
            dt_str = str(t.get("exit_datetime", ""))[:10]
            dt_obj = datetime.strptime(dt_str, "%Y-%m-%d")
            key = (dt_obj.year, dt_obj.month)
            monthly_data[key] = monthly_data.get(key, 0.0) + t["pnl_pct"]
        except (ValueError, TypeError):
            continue

    years = sorted(set(k[0] for k in monthly_data.keys())) if monthly_data else [2024]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    heatmap_z = []
    for y in years:
        row = [round(monthly_data.get((y, m), 0.0), 2) for m in range(1, 13)]
        heatmap_z.append(row)
    heatmap_z_json = json.dumps(heatmap_z)
    heatmap_y_json = json.dumps([str(y) for y in years])
    heatmap_x_json = json.dumps(month_names)

    # --- Diagnostic table ---
    diag_html = ""
    if tune_stats.get("trades", 0) > 0:
        def _decay(tune_val, hold_val):
            if tune_val and abs(tune_val) > 0.01:
                return f"{(1 - hold_val / tune_val) * 100:.0f}%"
            return "&mdash;"

        diag_rows = ""
        metrics = [
            ("Trades", "trades", "d"),
            ("Profit Factor", "pf", ".2f"),
            ("Win Rate", "wr_pct", ".1f"),
            ("Return %", "total_return_pct", ".1f"),
            ("Sharpe", "sharpe", ".2f"),
            ("Max DD %", "max_dd_pct", ".1f"),
        ]
        for label, key, fmt in metrics:
            tv = tune_stats.get(key, 0)
            hv = holdout_stats.get(key, 0) if holdout_stats else 0
            sv = survivor_stats.get(key, 0) if survivor_stats else 0
            hsv = holdout_survivor_stats.get(key, 0) if holdout_survivor_stats else 0
            diag_rows += f"""<tr>
                <td>{label}</td>
                <td>{tv:{fmt}}</td><td>{sv:{fmt}}</td>
                <td>{hv:{fmt}}</td><td>{hsv:{fmt}}</td>
                <td>{_decay(tv, hv)}</td>
            </tr>\n"""

        diag_html = f"""
        <h2>Performance Diagnostic: Tune vs Holdout</h2>
        <table>
        <thead><tr><th>Metric</th><th>Tune Universe</th><th>Tune Survivors</th>
          <th>Holdout Universe</th><th>Holdout Survivors</th><th>Decay</th></tr></thead>
        <tbody>{diag_rows}</tbody>
        </table>"""

    # --- Per-symbol table ---
    sym_rows = ""
    for sym in sorted_syms:
        if sym not in per_symbol:
            continue
        sd = per_symbol[sym]
        s = sd["stats"]
        hs = sd.get("holdout_stats", {})
        rob = robustness_data.get(sym, {})
        survived = sd.get("survived", False)
        surv_badge = '<span class="badge-pass">SURVIVOR</span>' if survived else ''
        sym_rows += f"""<tr>
            <td>{sym} {surv_badge}</td><td>{SECTOR_MAP.get(sym, 'Unknown')}</td>
            <td>{s.get('trades', 0)}</td><td>{s.get('pf', 0):.2f}</td>
            <td>{s.get('wr_pct', 0):.1f}%</td><td>{s.get('total_return_pct', 0):.1f}%</td>
            <td>{s.get('max_dd_pct', 0):.1f}%</td><td>{s.get('sharpe', 0):.2f}</td>
            <td>{hs.get('trades', 0)}</td><td>{hs.get('pf', 0):.2f}</td>
            <td>{hs.get('total_return_pct', 0):.1f}%</td>
            <td>{rob.get('composite', 0):.3f}</td>
        </tr>\n"""

    # --- Trade journal ---
    trade_rows = ""
    display_trades = all_trades[-200:] if len(all_trades) > 200 else all_trades
    for t in display_trades:
        pnl_val = t["pnl_pct"]
        pnl_class = "text-green" if pnl_val > 0 else "text-red"
        trade_rows += f"""<tr>
            <td>{t.get('symbol', '')}</td>
            <td>{str(t['entry_datetime'])[:16]}</td><td>{str(t['exit_datetime'])[:16]}</td>
            <td>${t['entry_price']:.2f}</td><td>${t['exit_price']:.2f}</td>
            <td class="{pnl_class}">{pnl_val:+.2f}%</td>
            <td>{t['bars_held']}</td><td>{t['exit_reason']}</td>
            <td>{t.get('direction', 'long')}</td>
        </tr>\n"""

    # --- Robustness table ---
    rob_rows = ""
    for sym in sorted_syms:
        rd = robustness_data.get(sym, {})
        if not rd:
            continue
        mc = rd.get("mc", {})
        comp = rd.get("composite", 0)
        comp_class = "text-green" if comp >= 0.4 else "text-red"
        rob_rows += f"""<tr>
            <td>{sym}</td>
            <td>{rd.get('mc_score', 0):.3f}</td>
            <td>{rd.get('noise_score', 0):.3f}</td>
            <td>{rd.get('stress_score', 0):.3f}</td>
            <td>{rd.get('sensitivity_score', 0):.3f}</td>
            <td class="{comp_class}">{comp:.3f}</td>
            <td>{mc.get('prob_profit', 0):.1%}</td>
            <td>{mc.get('p95_dd', 0):.1f}%</td>
        </tr>\n"""

    # --- Architecture description ---
    indicators = architecture.get('indicators', [])
    if indicators == ["UserStrategy"]:
        arch_desc = f"""<strong>Strategy:</strong> {run_info.get('concept', 'User Strategy')}<br>
        <strong>Mode:</strong> Direct entry/exit execution (user strategy file)<br>
        <strong>Timeframe:</strong> {architecture.get('exec_timeframe', 'N/A')}"""
    else:
        arch_desc = f"""<strong>Indicators:</strong> {', '.join(indicators)}<br>
        <strong>Exit Methods:</strong> {', '.join(architecture.get('exit_methods', []))}<br>
        <strong>Regime Model:</strong> {architecture.get('regime_model', 'N/A')}<br>
        <strong>Score Aggregation:</strong> {architecture.get('score_aggregation', 'N/A')}<br>
        <strong>Position Sizing:</strong> {architecture.get('position_sizing', 'N/A')}<br>
        <strong>Min Score:</strong> {architecture.get('min_score', 'N/A')}<br>
        <strong>Timeframe:</strong> {architecture.get('exec_timeframe', 'N/A')}"""

    # --- Parameter importance ---
    param_imp_rows = ""
    all_sens = {}
    for sym in sorted_syms:
        rd = robustness_data.get(sym, {})
        sens = rd.get("sensitivity", {})
        for pname, pdata in sens.items():
            if pname.startswith("_"):
                continue
            if pname not in all_sens:
                all_sens[pname] = {"stable_count": 0, "total": 0}
            all_sens[pname]["total"] += 1
            if pdata.get("stable", False):
                all_sens[pname]["stable_count"] += 1
    for pname, pdata in sorted(all_sens.items(),
                                key=lambda x: x[1]["stable_count"] / max(1, x[1]["total"])):
        stability_pct = pdata["stable_count"] / max(1, pdata["total"]) * 100
        bar_class = "text-green" if stability_pct >= 70 else "text-yellow" if stability_pct >= 40 else "text-red"
        param_imp_rows += f"""<tr>
            <td>{pname}</td>
            <td class="{bar_class}">{stability_pct:.0f}%</td>
            <td>{pdata['total']}</td>
        </tr>\n"""

    # --- Build stat cards ---
    def stat_card(value, label, fmt=".1f", color=None):
        if color is None:
            if isinstance(value, (int, float)):
                color = "green" if value > 0 else "red" if value < 0 else "blue"
            else:
                color = "blue"
        val_str = f"{value:{fmt}}" if isinstance(value, float) else str(value)
        return f'<div class="stat-card"><div class="stat-value text-{color}">{val_str}</div><div class="stat-label">{label}</div></div>'

    hs = headline_stats
    cards = [
        stat_card(hs.get('total_return_pct', 0), "Total Return %", "+.1f"),
        stat_card(hs.get('pf', 0), "Profit Factor", ".2f", "blue"),
        stat_card(hs.get('wr_pct', 0), "Win Rate %", ".1f", "blue"),
        stat_card(hs.get('trades', 0), "Total Trades", "d", "blue"),
        stat_card(-abs(hs.get('max_dd_pct', 0)), "Max Drawdown %", ".1f"),
        stat_card(hs.get('sharpe', 0), "Sharpe Ratio", ".2f"),
        stat_card(hs.get('sortino', 0), "Sortino Ratio", ".2f"),
        stat_card(hs.get('avg_bars_held', 0), "Avg Bars Held", ".1f", "blue"),
        stat_card(hs.get('avg_win', 0), "Avg Win %", "+.2f"),
        stat_card(hs.get('avg_loss', 0), "Avg Loss %", ".2f"),
        stat_card(len(survivor_syms), f"Survivors / {len(sorted_syms)}", "d", "blue"),
        stat_card(hs.get('edge_ratio', 0), "Edge Ratio", ".2f", "blue"),
    ]
    cards_html = "\n".join(cards)

    concept = run_info.get('concept', 'N/A')
    timestamp = run_info.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Strategy Report — {concept}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{
  --bg-primary: #0d1117; --bg-secondary: #161b22; --bg-tertiary: #21262d;
  --border: #30363d; --text-primary: #e6edf3; --text-secondary: #8b949e;
  --blue: #58a6ff; --green: #3fb950; --red: #f85149; --yellow: #d29922;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg-primary); color: var(--text-primary); font-family: var(--font); font-size: 14px; }}
.container {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
.text-green {{ color: var(--green); }} .text-red {{ color: var(--red); }}
.text-blue {{ color: var(--blue); }} .text-yellow {{ color: var(--yellow); }}

/* Header */
.header {{ background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); border-bottom: 2px solid var(--blue); padding: 32px 24px; margin-bottom: 24px; }}
.header h1 {{ color: var(--blue); font-size: 24px; margin-bottom: 8px; }}
.header .subtitle {{ color: var(--text-secondary); font-size: 13px; }}
.header .headline {{ display: inline-block; margin-top: 12px; padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; }}
.headline-holdout {{ background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid rgba(63,185,80,0.3); }}
.headline-tune {{ background: rgba(210,153,34,0.15); color: var(--yellow); border: 1px solid rgba(210,153,34,0.3); }}

/* Stats grid */
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; margin: 20px 0; }}
.stat-card {{ background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 16px 12px; text-align: center; }}
.stat-value {{ font-size: 22px; font-weight: 700; line-height: 1.2; }}
.stat-label {{ font-size: 11px; color: var(--text-secondary); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}

/* Tabs */
.tabs {{ display: flex; gap: 2px; margin: 24px 0 0; border-bottom: 1px solid var(--border); }}
.tab {{ padding: 10px 20px; background: transparent; border: none; color: var(--text-secondary); font-size: 13px; cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.15s; }}
.tab:hover {{ color: var(--text-primary); }}
.tab.active {{ color: var(--blue); border-bottom-color: var(--blue); }}
.tab-page {{ display: none; padding: 24px 0; }}
.tab-page.active {{ display: block; }}

/* Tables */
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
thead th {{ background: var(--bg-tertiary); color: var(--blue); padding: 10px 12px; text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 0.3px; cursor: pointer; white-space: nowrap; border-bottom: 1px solid var(--border); }}
tbody td {{ padding: 8px 12px; border-bottom: 1px solid var(--bg-tertiary); font-size: 13px; }}
tbody tr:hover {{ background: rgba(88,166,255,0.04); }}

/* Charts */
.chart-box {{ background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin: 16px 0; }}

/* Exit breakdown */
.exit-bar {{ display: flex; align-items: center; margin: 6px 0; gap: 8px; }}
.exit-label {{ width: 140px; font-size: 13px; color: var(--text-secondary); text-align: right; }}
.exit-fill {{ background: var(--blue); height: 20px; border-radius: 4px; min-width: 2px; transition: width 0.3s; }}
.exit-pct {{ font-size: 12px; color: var(--text-secondary); white-space: nowrap; }}

/* Arch box */
.arch-box {{ background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px; line-height: 1.8; }}

/* Badges */
.badge-pass {{ background: rgba(63,185,80,0.15); color: var(--green); padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}

h2 {{ color: var(--text-primary); font-size: 18px; margin: 28px 0 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
h3 {{ color: var(--text-secondary); font-size: 14px; margin: 16px 0 8px; }}
.footer {{ text-align: center; color: var(--text-secondary); margin-top: 48px; padding: 20px; font-size: 12px; border-top: 1px solid var(--border); }}
</style>
</head>
<body>

<div class="header">
  <h1>{concept}</h1>
  <div class="subtitle">Generated {timestamp} &bull; {len(sorted_syms)} symbols &bull; {len(all_trades)} tune trades &bull; {len(holdout_trades)} holdout trades</div>
  <div class="headline {'headline-holdout' if use_holdout else 'headline-tune'}">{headline_label}</div>
</div>

<div class="container">

<div class="stats-grid">{cards_html}</div>

<div class="tabs">
  <div class="tab active" onclick="showTab(0)">Summary</div>
  <div class="tab" onclick="showTab(1)">Equity &amp; Returns</div>
  <div class="tab" onclick="showTab(2)">Per-Symbol</div>
  <div class="tab" onclick="showTab(3)">Trade Journal</div>
  <div class="tab" onclick="showTab(4)">Robustness</div>
  <div class="tab" onclick="showTab(5)">Optimization</div>
</div>

<!-- Summary -->
<div class="tab-page active" id="page0">
{diag_html}
<h2>Exit Reason Breakdown</h2>
{exit_items}
<h2>Architecture</h2>
<div class="arch-box">{arch_desc}</div>
</div>

<!-- Equity & Returns -->
<div class="tab-page" id="page1">
<div class="chart-box" id="equity-chart" style="height:420px;"></div>
<div class="chart-box" id="dd-chart" style="height:280px;"></div>
<h2>Monthly Returns</h2>
<div class="chart-box" id="heatmap-chart" style="height:320px;"></div>
</div>

<!-- Per-Symbol -->
<div class="tab-page" id="page2">
<h2>Per-Symbol Results</h2>
<table id="sym-table">
<thead><tr>
  <th onclick="sortTable('sym-table',0)">Symbol</th>
  <th onclick="sortTable('sym-table',1)">Sector</th>
  <th onclick="sortTable('sym-table',2)">Tune Trades</th>
  <th onclick="sortTable('sym-table',3)">Tune PF</th>
  <th onclick="sortTable('sym-table',4)">Win Rate</th>
  <th onclick="sortTable('sym-table',5)">Tune Return</th>
  <th onclick="sortTable('sym-table',6)">Max DD</th>
  <th onclick="sortTable('sym-table',7)">Sharpe</th>
  <th onclick="sortTable('sym-table',8)">Holdout Trades</th>
  <th onclick="sortTable('sym-table',9)">Holdout PF</th>
  <th onclick="sortTable('sym-table',10)">Holdout Return</th>
  <th onclick="sortTable('sym-table',11)">Robustness</th>
</tr></thead>
<tbody>{sym_rows}</tbody>
</table>
</div>

<!-- Trade Journal -->
<div class="tab-page" id="page3">
<h2>Trade Journal (last 200)</h2>
<table id="trade-table">
<thead><tr>
  <th>Symbol</th><th>Entry</th><th>Exit</th><th>Entry $</th><th>Exit $</th>
  <th onclick="sortTable('trade-table',5)">PnL %</th><th>Bars</th><th>Exit Reason</th><th>Direction</th>
</tr></thead>
<tbody>{trade_rows}</tbody>
</table>
</div>

<!-- Robustness -->
<div class="tab-page" id="page4">
<h2>Robustness Gauntlet</h2>
<table>
<thead><tr>
  <th>Symbol</th><th>MC Score</th><th>Noise Score</th><th>Stress Score</th>
  <th>Sensitivity</th><th>Composite</th><th>MC Prob Profit</th><th>MC P95 DD</th>
</tr></thead>
<tbody>{rob_rows}</tbody>
</table>
</div>

<!-- Optimization -->
<div class="tab-page" id="page5">
<h2>Parameter Sensitivity</h2>
<table>
<thead><tr><th>Parameter</th><th>Stability</th><th>Symbols Tested</th></tr></thead>
<tbody>{param_imp_rows if param_imp_rows else '<tr><td colspan="3" style="text-align:center;color:var(--text-secondary);">No parameter sensitivity data (strategy mode without tunable params)</td></tr>'}</tbody>
</table>
</div>

</div>

<div class="footer">Optuna Screener Pipeline &bull; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

<script>
function showTab(idx) {{
  document.querySelectorAll('.tab-page').forEach((p, i) => p.classList.toggle('active', i === idx));
  document.querySelectorAll('.tab').forEach((t, i) => t.classList.toggle('active', i === idx));
  if (idx === 1) setTimeout(() => {{
    ['equity-chart','dd-chart','heatmap-chart'].forEach(id => {{
      var el = document.getElementById(id);
      if (el && el.data) Plotly.Plots.resize(el);
    }});
  }}, 50);
}}

function sortTable(tableId, col) {{
  var table = document.getElementById(tableId);
  var tbody = table.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var dir = table.dataset.sortDir === 'asc' ? 'desc' : 'asc';
  table.dataset.sortDir = dir;
  rows.sort(function(a, b) {{
    var va = a.cells[col].textContent.replace(/[%$,]/g,'').trim();
    var vb = b.cells[col].textContent.replace(/[%$,]/g,'').trim();
    var na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return dir === 'asc' ? na - nb : nb - na;
    return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

var darkLayout = {{
  paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
  font: {{ color: '#e6edf3', size: 12 }},
  xaxis: {{ gridcolor: '#21262d', linecolor: '#30363d', zeroline: false }},
  yaxis: {{ gridcolor: '#21262d', linecolor: '#30363d', zeroline: false }},
  margin: {{ l: 60, r: 20, t: 40, b: 40 }}, showlegend: true,
  legend: {{ bgcolor: 'rgba(0,0,0,0)', font: {{ color: '#8b949e' }} }}
}};

var eqDates = {eq_dates_json};
var eqVals = {eq_vals_json};
if (eqDates.length > 0) {{
  Plotly.newPlot('equity-chart', [{bench_trace}
    {{ x: eqDates, y: eqVals, type: 'scatter', mode: 'lines', name: 'Portfolio', line: {{ color: '#58a6ff', width: 2 }} }}
  ], Object.assign({{}}, darkLayout, {{ title: 'Portfolio Equity Curve', yaxis: {{ title: 'Equity ($)', gridcolor: '#21262d' }} }}), {{ responsive: true }});

  Plotly.newPlot('dd-chart', [{{
    x: eqDates, y: {dd_json}, type: 'scatter', mode: 'lines', fill: 'tozeroy', name: 'Drawdown',
    line: {{ color: '#f85149', width: 1 }}, fillcolor: 'rgba(248,81,73,0.15)'
  }}], Object.assign({{}}, darkLayout, {{ title: 'Drawdown', yaxis: {{ title: 'DD (%)', gridcolor: '#21262d' }} }}), {{ responsive: true }});
}} else {{
  document.getElementById('equity-chart').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#8b949e;">No equity curve data available</div>';
  document.getElementById('dd-chart').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#8b949e;">No drawdown data available</div>';
}}

Plotly.newPlot('heatmap-chart', [{{
  z: {heatmap_z_json}, x: {heatmap_x_json}, y: {heatmap_y_json},
  type: 'heatmap', colorscale: [[0,'#f85149'],[0.5,'#161b22'],[1,'#3fb950']],
  showscale: true, colorbar: {{ title: 'PnL %', tickfont: {{ color: '#8b949e' }} }}
}}], Object.assign({{}}, darkLayout, {{ title: 'Monthly Returns', yaxis: {{ autorange: 'reversed' }} }}), {{ responsive: true }});
</script>
</body>
</html>"""

    report_path = od / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"HTML report saved: {report_path}")
    return str(report_path)
