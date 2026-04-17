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
    never saw) whenever holdout data exists, with a TUNE-vs-HOLDOUT diagnostic
    table so the reader can see performance decay.
    """
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    holdout_stats = results.get("holdout_universe_stats", {}) or {}
    tune_stats = results.get("portfolio_stats", {}) or {}
    use_holdout = holdout_stats.get("trades", 0) > 0
    stats = holdout_stats if use_holdout else tune_stats
    headline_label = "TRUE OOS HOLDOUT" if use_holdout else "TUNE WINDOW (biased)"
    all_trades = results.get("all_trades", [])

    # Exit reason breakdown
    exit_counts = {}
    for t in all_trades:
        r = t.get("exit_reason", "unknown")
        exit_counts[r] = exit_counts.get(r, 0) + 1
    _target_n = exit_counts.get("fixed_target", 0)
    _atr_n = exit_counts.get("fixed_stop", 0)
    _trail_n = exit_counts.get("trailing_stop", 0)
    _regime_n = exit_counts.get("regime_exit", 0)
    _time_n = exit_counts.get("time_exit", 0)
    _exit_total = max(1, sum(exit_counts.values()))
    exit_html = f"""
    <div style="margin:20px;padding:15px;border:1px solid #30363d;background:#161b22;">
      <h2>Exit Reason Breakdown</h2>
      <ul>
        <li>fixed_target: {_target_n} ({_target_n/_exit_total*100:.1f}%)</li>
        <li>fixed_stop: {_atr_n} ({_atr_n/_exit_total*100:.1f}%)</li>
        <li>trailing_stop: {_trail_n} ({_trail_n/_exit_total*100:.1f}%)</li>
        <li>regime_exit: {_regime_n} ({_regime_n/_exit_total*100:.1f}%)</li>
        <li>time_exit: {_time_n} ({_time_n/_exit_total*100:.1f}%)</li>
      </ul>
    </div>
    """

    header_banner = f"""
    <div style="background:#1a3a52;color:white;padding:20px;text-align:center;font-size:24px;">
      Optuna Screener Results &mdash; {headline_label}
    </div>
    """

    if holdout_stats.get("trades", 0) > 0 and tune_stats.get("trades", 0) > 0:
        diag_html = f"""
        <div style="margin:20px;">
        <h2>Diagnostic: TUNE vs HOLDOUT</h2>
        <table border="1" style="border-collapse:collapse;width:100%;">
          <tr><th>Metric</th><th>TUNE (biased)</th><th>HOLDOUT (true OOS)</th><th>Decay</th></tr>
          <tr><td>Trades</td><td>{tune_stats.get('trades', 0)}</td>
              <td>{holdout_stats.get('trades', 0)}</td><td>&mdash;</td></tr>
          <tr><td>PF</td><td>{tune_stats.get('pf', 0):.2f}</td>
              <td>{holdout_stats.get('pf', 0):.2f}</td>
              <td>{(1 - holdout_stats.get('pf', 0) / max(tune_stats.get('pf', 1), 0.01)) * 100:.0f}%</td></tr>
          <tr><td>Sharpe</td><td>{tune_stats.get('sharpe', 0):.2f}</td>
              <td>{holdout_stats.get('sharpe', 0):.2f}</td>
              <td>{(1 - holdout_stats.get('sharpe', 0) / max(tune_stats.get('sharpe', 1), 0.01)) * 100:.0f}%</td></tr>
          <tr><td>Win %</td><td>{tune_stats.get('wr_pct', 0):.1f}%</td>
              <td>{holdout_stats.get('wr_pct', 0):.1f}%</td><td>&mdash;</td></tr>
          <tr><td>Return</td><td>{tune_stats.get('total_return_pct', 0):.1f}%</td>
              <td>{holdout_stats.get('total_return_pct', 0):.1f}%</td><td>&mdash;</td></tr>
        </table>
        </div>
        """
    else:
        diag_html = ""
    per_symbol = results.get("per_symbol", {})
    sorted_syms = results.get("sorted_syms", [])
    equity_dates = results.get("equity_dates", [])
    equity_values = results.get("equity_values", [])
    benchmark = results.get("benchmark")

    eq_dates_json = json.dumps(equity_dates)
    eq_vals_json = json.dumps([round(v, 2) for v in equity_values])

    bench_trace = ""
    if benchmark:
        bench_dates = json.dumps([str(d) for d in benchmark["dates"]])
        bench_vals = json.dumps([round(v, 2) for v in benchmark["equity"]])
        bench_trace = f"""
        {{
            x: {bench_dates},
            y: {bench_vals},
            type: 'scatter',
            mode: 'lines',
            name: 'SPY Benchmark',
            line: {{color: '#888888', dash: 'dot'}}
        }},"""

    dd_values = []
    peak_eq = 10000.0
    for v in equity_values:
        if v > peak_eq:
            peak_eq = v
        dd = (peak_eq - v) / peak_eq * 100.0
        dd_values.append(round(-dd, 2))
    dd_json = json.dumps(dd_values)

    # Monthly returns heatmap
    monthly_data = {}
    for t in all_trades:
        try:
            dt_str = t.get("exit_datetime", "")[:10]
            dt_obj = datetime.strptime(dt_str, "%Y-%m-%d")
            key = (dt_obj.year, dt_obj.month)
            monthly_data[key] = monthly_data.get(key, 0.0) + t["pnl_pct"]
        except (ValueError, TypeError):
            continue

    years = sorted(set(k[0] for k in monthly_data.keys())) if monthly_data else [2024]
    months = list(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    heatmap_z = []
    for y in years:
        row = []
        for m in months:
            row.append(round(monthly_data.get((y, m), 0.0), 2))
        heatmap_z.append(row)
    heatmap_z_json = json.dumps(heatmap_z)
    heatmap_y_json = json.dumps([str(y) for y in years])
    heatmap_x_json = json.dumps(month_names)

    sym_rows = ""
    for sym in sorted_syms:
        if sym not in per_symbol:
            continue
        s = per_symbol[sym]["stats"]
        rob = robustness_data.get(sym, {})
        sym_rows += f"""<tr>
            <td>{sym}</td><td>{SECTOR_MAP.get(sym, 'Unknown')}</td>
            <td>{s.get('trades', 0)}</td><td>{s.get('pf', 0):.2f}</td>
            <td>{s.get('wr_pct', 0):.1f}%</td><td>{s.get('total_return_pct', 0):.1f}%</td>
            <td>{s.get('max_dd_pct', 0):.1f}%</td><td>{s.get('sharpe', 0):.2f}</td>
            <td>{rob.get('composite', 0):.3f}</td>
        </tr>\n"""

    trade_rows = ""
    display_trades = all_trades[-200:] if len(all_trades) > 200 else all_trades
    for t in display_trades:
        pnl_class = "positive" if t["pnl_pct"] > 0 else "negative"
        trade_rows += f"""<tr class="{pnl_class}">
            <td>{t.get('symbol', '')}</td>
            <td>{t['entry_datetime'][:19]}</td><td>{t['exit_datetime'][:19]}</td>
            <td>{t['entry_price']:.2f}</td><td>{t['exit_price']:.2f}</td>
            <td>{t['pnl_pct']:.2f}%</td><td>{t['bars_held']}</td>
            <td>{t['exit_reason']}</td><td>{t['entry_regime']}</td>
        </tr>\n"""

    rob_rows = ""
    for sym in sorted_syms:
        rd = robustness_data.get(sym, {})
        if not rd:
            continue
        mc = rd.get("mc", {})
        rob_rows += f"""<tr>
            <td>{sym}</td>
            <td>{rd.get('mc_score', 0):.3f}</td>
            <td>{rd.get('noise_score', 0):.3f}</td>
            <td>{rd.get('stress_score', 0):.3f}</td>
            <td>{rd.get('sensitivity_score', 0):.3f}</td>
            <td>{rd.get('composite', 0):.3f}</td>
            <td>{mc.get('prob_profit', 0):.1%}</td>
            <td>{mc.get('p95_dd', 0):.1f}%</td>
        </tr>\n"""

    # Validation tab rows
    has_validation = False
    val_rows = ""
    for sym in sorted_syms:
        rd = robustness_data.get(sym, {})
        if not rd:
            continue
        smc_frac = rd.get("synthetic_mc_profitable_frac")
        dsr_val = rd.get("dsr")
        pbo_val = rd.get("pbo")
        if smc_frac is not None or dsr_val is not None or pbo_val is not None:
            has_validation = True
        smc_display = f"{smc_frac*100:.1f}%" if smc_frac is not None else "&mdash;"
        smc_pass = rd.get("synthetic_mc_pass")
        smc_badge = ""
        if smc_pass is True:
            smc_badge = ' <span style="color:#3fb950;">PASS</span>'
        elif smc_pass is False:
            smc_badge = ' <span style="color:#f85149;">FAIL</span>'
        dsr_display = f"{dsr_val:.4f}" if dsr_val is not None else "&mdash;"
        pbo_display = f"{pbo_val:.4f}" if pbo_val is not None else "&mdash;"
        val_rows += f"""<tr>
            <td>{sym}</td>
            <td>{smc_display}{smc_badge}</td>
            <td>{dsr_display}</td>
            <td>{pbo_display}</td>
        </tr>\n"""

    arch_desc = f"""
    <strong>Indicators:</strong> {', '.join(architecture.get('indicators', []))}<br>
    <strong>Exit Methods:</strong> {', '.join(architecture.get('exit_methods', []))}<br>
    <strong>Regime Model:</strong> {architecture.get('regime_model', 'N/A')}<br>
    <strong>Score Aggregation:</strong> {architecture.get('score_aggregation', 'N/A')}<br>
    <strong>Position Sizing:</strong> {architecture.get('position_sizing', 'N/A')}<br>
    <strong>Min Score:</strong> {architecture.get('min_score', 'N/A')}<br>
    <strong>Timeframe:</strong> {architecture.get('exec_timeframe', 'N/A')}
    """

    param_imp_rows = ""
    all_sens = {}
    for sym in sorted_syms:
        rd = robustness_data.get(sym, {})
        sens = rd.get("sensitivity", {})
        for pname, pdata in sens.items():
            if pname not in all_sens:
                all_sens[pname] = {"stable_count": 0, "total": 0}
            all_sens[pname]["total"] += 1
            if pdata.get("stable", False):
                all_sens[pname]["stable_count"] += 1
    for pname, pdata in sorted(all_sens.items(), key=lambda x: x[1]["stable_count"] / max(1, x[1]["total"])):
        stability_pct = pdata["stable_count"] / max(1, pdata["total"]) * 100
        param_imp_rows += f"""<tr>
            <td>{pname}</td>
            <td>{stability_pct:.0f}%</td>
            <td>{pdata['total']}</td>
        </tr>\n"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Optuna Screener Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', Tahoma, sans-serif; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #58a6ff; font-size: 28px; margin-bottom: 10px; }}
  h2 {{ color: #58a6ff; font-size: 22px; margin: 30px 0 15px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  h3 {{ color: #8b949e; font-size: 16px; margin: 15px 0 10px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }}
  .stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }}
  .stat-card .value {{ font-size: 24px; font-weight: bold; color: #58a6ff; }}
  .stat-card .label {{ font-size: 12px; color: #8b949e; margin-top: 5px; }}
  .stat-card.positive .value {{ color: #3fb950; }}
  .stat-card.negative .value {{ color: #f85149; }}
  .arch-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 15px 0; line-height: 1.8; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: #161b22; border-radius: 8px; overflow: hidden; }}
  th {{ background: #21262d; color: #58a6ff; padding: 10px 12px; text-align: left; font-size: 13px; cursor: pointer; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; font-size: 13px; }}
  tr:hover {{ background: #1c2128; }}
  tr.positive td:nth-child(6) {{ color: #3fb950; }}
  tr.negative td:nth-child(6) {{ color: #f85149; }}
  .chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin: 15px 0; }}
  .tab-container {{ display: flex; gap: 5px; margin: 20px 0 0; flex-wrap: wrap; }}
  .tab {{ padding: 10px 20px; background: #21262d; border: 1px solid #30363d; border-bottom: none;
           border-radius: 8px 8px 0 0; cursor: pointer; color: #8b949e; font-size: 14px; }}
  .tab.active {{ background: #161b22; color: #58a6ff; border-color: #58a6ff; }}
  .tab-page {{ display: none; border: 1px solid #30363d; border-radius: 0 8px 8px 8px; padding: 20px; background: #161b22; }}
  .tab-page.active {{ display: block; }}
  .footer {{ text-align: center; color: #484f58; margin-top: 40px; padding: 20px; font-size: 12px; }}
</style>
</head>
<body>
{header_banner}
{diag_html}
{exit_html}
<div class="container">
<h1>Optuna Screener Report</h1>
<p style="color:#8b949e;">Generated: {run_info.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
| Concept: {run_info.get('concept', 'N/A')} | Symbols: {len(sorted_syms)}</p>

<div class="tab-container">
  <div class="tab active" onclick="showTab(0)">Executive Summary</div>
  <div class="tab" onclick="showTab(1)">Equity &amp; Returns</div>
  <div class="tab" onclick="showTab(2)">Per-Symbol</div>
  <div class="tab" onclick="showTab(3)">Trade Journal</div>
  <div class="tab" onclick="showTab(4)">Optimization</div>
  <div class="tab" onclick="showTab(5)">Robustness</div>
  <div class="tab" onclick="showTab(6)">Validation</div>
</div>

<!-- PAGE 1: Executive Summary -->
<div class="tab-page active" id="page0">
<h2>Executive Summary</h2>
<div class="stats-grid">
  <div class="stat-card {'positive' if stats.get('total_return_pct', 0) > 0 else 'negative'}">
    <div class="value">{stats.get('total_return_pct', 0):.1f}%</div><div class="label">Total Return</div>
  </div>
  <div class="stat-card"><div class="value">{stats.get('pf', 0):.2f}</div><div class="label">Profit Factor</div></div>
  <div class="stat-card"><div class="value">{stats.get('wr_pct', 0):.1f}%</div><div class="label">Win Rate</div></div>
  <div class="stat-card"><div class="value">{stats.get('trades', 0)}</div><div class="label">Total Trades</div></div>
  <div class="stat-card negative"><div class="value">{stats.get('max_dd_pct', 0):.1f}%</div><div class="label">Max Drawdown</div></div>
  <div class="stat-card"><div class="value">{stats.get('sharpe', 0):.2f}</div><div class="label">Sharpe Ratio</div></div>
  <div class="stat-card"><div class="value">{stats.get('sortino', 0):.2f}</div><div class="label">Sortino Ratio</div></div>
  <div class="stat-card"><div class="value">{stats.get('edge_ratio', 0):.2f}</div><div class="label">Edge Ratio</div></div>
  <div class="stat-card"><div class="value">{stats.get('avg_bars_held', 0):.1f}</div><div class="label">Avg Bars Held</div></div>
  <div class="stat-card positive"><div class="value">{stats.get('avg_win', 0):.2f}%</div><div class="label">Avg Win</div></div>
  <div class="stat-card negative"><div class="value">{stats.get('avg_loss', 0):.2f}%</div><div class="label">Avg Loss</div></div>
  <div class="stat-card"><div class="value">{len(sorted_syms)}</div><div class="label">Symbols</div></div>
</div>
<h3>Architecture</h3>
<div class="arch-box">{arch_desc}</div>
</div>

<!-- PAGE 2: Equity & Returns -->
<div class="tab-page" id="page1">
<h2>Equity Curve</h2>
<div class="chart-container" id="equity-chart" style="height:450px;"></div>
<h2>Drawdown</h2>
<div class="chart-container" id="dd-chart" style="height:300px;"></div>
<h2>Monthly Returns Heatmap</h2>
<div class="chart-container" id="heatmap-chart" style="height:350px;"></div>
</div>

<!-- PAGE 3: Per-Symbol -->
<div class="tab-page" id="page2">
<h2>Per-Symbol Results</h2>
<table id="sym-table">
<thead><tr>
  <th onclick="sortTable('sym-table',0)">Symbol</th>
  <th onclick="sortTable('sym-table',1)">Sector</th>
  <th onclick="sortTable('sym-table',2)">Trades</th>
  <th onclick="sortTable('sym-table',3)">PF</th>
  <th onclick="sortTable('sym-table',4)">Win Rate</th>
  <th onclick="sortTable('sym-table',5)">Return</th>
  <th onclick="sortTable('sym-table',6)">Max DD</th>
  <th onclick="sortTable('sym-table',7)">Sharpe</th>
  <th onclick="sortTable('sym-table',8)">Robustness</th>
</tr></thead>
<tbody>{sym_rows}</tbody>
</table>
</div>

<!-- PAGE 4: Trade Journal -->
<div class="tab-page" id="page3">
<h2>Trade Journal (last 200)</h2>
<table id="trade-table">
<thead><tr>
  <th>Symbol</th><th>Entry</th><th>Exit</th><th>Entry$</th><th>Exit$</th>
  <th>PnL%</th><th>Bars</th><th>Exit Reason</th><th>Regime</th>
</tr></thead>
<tbody>{trade_rows}</tbody>
</table>
</div>

<!-- PAGE 5: Optimization -->
<div class="tab-page" id="page4">
<h2>Parameter Importance</h2>
<table>
<thead><tr><th>Parameter</th><th>Stability</th><th>Symbols Tested</th></tr></thead>
<tbody>{param_imp_rows}</tbody>
</table>
</div>

<!-- PAGE 6: Robustness -->
<div class="tab-page" id="page5">
<h2>Robustness Gauntlet Results</h2>
<table>
<thead><tr>
  <th>Symbol</th><th>MC Score</th><th>Noise Score</th><th>Stress Score</th>
  <th>Sensitivity</th><th>Composite</th><th>MC Prob Profit</th><th>MC P95 DD</th>
</tr></thead>
<tbody>{rob_rows}</tbody>
</table>
</div>

<!-- PAGE 7: Validation -->
<div class="tab-page" id="page6">
<h2>Validation Suite</h2>
{"<p style='color:#8b949e;'>Synthetic MC, Deflated Sharpe Ratio, and Probability of Backtest Overfitting results per symbol.</p>" if has_validation else "<p style='color:#8b949e;'>No validation tests were enabled for this run. Enable them in <code>apex_config.json</code> under <code>validation</code>.</p>"}
<table>
<thead><tr>
  <th>Symbol</th><th>Synthetic MC Pass Rate</th><th>DSR</th><th>PBO</th>
</tr></thead>
<tbody>{val_rows}</tbody>
</table>
</div>

</div><!-- end container -->

<div class="footer">
Optuna Screener Pipeline | Report generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>

<script>
function showTab(idx) {{
  document.querySelectorAll('.tab-page').forEach((p, i) => {{
    p.classList.toggle('active', i === idx);
  }});
  document.querySelectorAll('.tab').forEach((t, i) => {{
    t.classList.toggle('active', i === idx);
  }});
  if (idx === 1) {{
    setTimeout(() => {{
      Plotly.Plots.resize('equity-chart');
      Plotly.Plots.resize('dd-chart');
      Plotly.Plots.resize('heatmap-chart');
    }}, 100);
  }}
}}

function sortTable(tableId, col) {{
  var table = document.getElementById(tableId);
  var tbody = table.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var dir = table.dataset.sortDir === 'asc' ? 'desc' : 'asc';
  table.dataset.sortDir = dir;
  rows.sort(function(a, b) {{
    var va = a.cells[col].textContent.replace('%','').trim();
    var vb = b.cells[col].textContent.replace('%','').trim();
    var na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) {{
      return dir === 'asc' ? na - nb : nb - na;
    }}
    return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

var darkLayout = {{
  paper_bgcolor: '#161b22',
  plot_bgcolor: '#161b22',
  font: {{ color: '#c9d1d9' }},
  xaxis: {{ gridcolor: '#21262d', linecolor: '#30363d' }},
  yaxis: {{ gridcolor: '#21262d', linecolor: '#30363d' }},
  margin: {{ l: 60, r: 30, t: 40, b: 50 }},
}};

Plotly.newPlot('equity-chart', [
  {bench_trace}
  {{
    x: {eq_dates_json},
    y: {eq_vals_json},
    type: 'scatter',
    mode: 'lines',
    name: 'Portfolio',
    line: {{ color: '#58a6ff', width: 2 }}
  }}
], Object.assign({{}}, darkLayout, {{
  title: 'Portfolio Equity Curve',
  yaxis: {{ title: 'Equity ($)', gridcolor: '#21262d' }}
}}), {{ responsive: true }});

Plotly.newPlot('dd-chart', [{{
  x: {eq_dates_json},
  y: {dd_json},
  type: 'scatter',
  mode: 'lines',
  fill: 'tozeroy',
  name: 'Drawdown',
  line: {{ color: '#f85149' }},
  fillcolor: 'rgba(248,81,73,0.2)'
}}], Object.assign({{}}, darkLayout, {{
  title: 'Drawdown',
  yaxis: {{ title: 'Drawdown (%)', gridcolor: '#21262d' }}
}}), {{ responsive: true }});

Plotly.newPlot('heatmap-chart', [{{
  z: {heatmap_z_json},
  x: {heatmap_x_json},
  y: {heatmap_y_json},
  type: 'heatmap',
  colorscale: [
    [0, '#f85149'],
    [0.5, '#161b22'],
    [1, '#3fb950']
  ],
  showscale: true,
  colorbar: {{ title: 'Return %', tickfont: {{ color: '#c9d1d9' }} }}
}}], Object.assign({{}}, darkLayout, {{
  title: 'Monthly Returns',
  yaxis: {{ autorange: 'reversed' }}
}}), {{ responsive: true }});
</script>
</body>
</html>"""

    report_path = od / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"HTML report saved: {report_path}")
    return str(report_path)
