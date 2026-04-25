"""
HTMLReporter: Generates a rich, interactive HTML evaluation report.

The report includes:
  - Executive summary with leaderboard table
  - Per-scenario radar charts (Plotly)
  - Model comparison bar charts
  - Per-sample prediction browser
  - Side-by-side model output tables
  - Metric correlation heatmaps
"""

from __future__ import annotations

import base64
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from vlm_eval.core.config import EvalConfig
from vlm_eval.core.scenario import ScenarioResult


REPORT_CSS = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', -apple-system, sans-serif; background: #f0f2f5; color: #1a1a2e; }
  .header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white; padding: 2.5rem 3rem; }
  .header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
  .header p  { color: #a0aec0; font-size: 0.95rem; }
  .badge { display: inline-block; background: #e94560; color: white;
           border-radius: 12px; padding: 2px 10px; font-size: 0.75rem;
           margin-left: 8px; vertical-align: middle; }
  .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
  .card { background: white; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
          margin-bottom: 1.5rem; overflow: hidden; }
  .card-header { background: #f7fafc; border-bottom: 1px solid #e2e8f0;
                 padding: 1rem 1.5rem; font-weight: 600; font-size: 1rem;
                 display: flex; align-items: center; gap: 0.5rem; }
  .card-body { padding: 1.5rem; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th { background: #f7fafc; padding: 0.75rem 1rem; text-align: left;
       font-weight: 600; border-bottom: 2px solid #e2e8f0; }
  td { padding: 0.7rem 1rem; border-bottom: 1px solid #f0f0f0; }
  tr:hover td { background: #f7fafc; }
  .metric { font-weight: 600; color: #2b6cb0; }
  .score-bar-bg { background: #e2e8f0; border-radius: 4px; height: 8px; width: 100px;
                  display: inline-block; vertical-align: middle; margin-left: 8px; }
  .score-bar { background: linear-gradient(90deg, #4299e1, #667eea);
               border-radius: 4px; height: 8px; }
  .tag { display: inline-block; background: #ebf8ff; color: #2b6cb0;
         border-radius: 8px; padding: 1px 8px; font-size: 0.75rem; margin: 1px; }
  .tag.anomaly { background: #fff5f5; color: #c53030; }
  .tag.scene   { background: #f0fff4; color: #276749; }
  .tag.safety  { background: #fffaf0; color: #c05621; }
  .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
  .stat-card { background: linear-gradient(135deg, #667eea, #764ba2);
               color: white; border-radius: 10px; padding: 1.2rem 1.5rem; text-align: center; }
  .stat-card .number { font-size: 2rem; font-weight: 700; }
  .stat-card .label  { font-size: 0.8rem; opacity: 0.85; margin-top: 2px; }
  .pred-box { background: #f7fafc; border-left: 4px solid #4299e1;
              border-radius: 4px; padding: 0.75rem 1rem; font-size: 0.88rem;
              font-family: monospace; line-height: 1.5; color: #2d3748; }
  .ref-box  { background: #f0fff4; border-left: 4px solid #48bb78;
              border-radius: 4px; padding: 0.75rem 1rem; font-size: 0.88rem;
              font-family: monospace; line-height: 1.5; color: #2d3748; }
  .tabs { display: flex; border-bottom: 2px solid #e2e8f0; margin-bottom: 1rem; }
  .tab-btn { padding: 0.6rem 1.2rem; cursor: pointer; border: none; background: none;
             font-size: 0.9rem; color: #718096; border-bottom: 2px solid transparent;
             margin-bottom: -2px; transition: all 0.2s; }
  .tab-btn.active { color: #2b6cb0; border-bottom-color: #2b6cb0; font-weight: 600; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  footer { text-align: center; color: #a0aec0; font-size: 0.8rem; padding: 2rem; }
  @media (max-width: 768px) { .grid-2, .grid-3, .stat-grid { grid-template-columns: 1fr; } }
</style>
"""

REPORT_JS = """
<script>
function switchTab(tabGroup, tabId) {
  document.querySelectorAll('[data-group="' + tabGroup + '"].tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('[data-group="' + tabGroup + '"].tab-content').forEach(c => c.classList.remove('active'));
  document.querySelector('[data-group="' + tabGroup + '"][data-tab="' + tabId + '"].tab-btn').classList.add('active');
  document.querySelector('[data-group="' + tabGroup + '"][data-tab="' + tabId + '"].tab-content').classList.add('active');
}
</script>
"""


class HTMLReporter:
    """Generates a standalone HTML report from evaluation results."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.output_path = config.output_path

    def generate(
        self,
        all_results: Dict[str, List[ScenarioResult]],
    ) -> Path:
        """Generate the HTML report and return its path."""
        report_path = self.output_path / "eval_report.html"

        html = self._build_html(all_results)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Report generated: {report_path}")
        return report_path

    def _build_html(self, all_results: Dict[str, List[ScenarioResult]]) -> str:
        """Assemble the full HTML document."""
        model_names = list(all_results.keys())
        scenario_names = [s.name for s in self.config.scenarios]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Aggregate stats
        total_samples = sum(
            sr.n_samples
            for results in all_results.values()
            for sr in results
        )
        total_scenarios = len(scenario_names)
        total_models = len(model_names)
        all_metrics = set()
        for results in all_results.values():
            for sr in results:
                all_metrics.update(sr.aggregate_scores().keys())

        plotly_charts = self._build_plotly_charts(all_results)

        body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{self.config.suite_name} — VLM Eval Report</title>
  {REPORT_CSS}
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <h1>🔬 {self.config.suite_name}
    <span class="badge">VLM Eval v{self.config.version}</span>
  </h1>
  <p>Vision-Language Model Evaluation Report · Generated {timestamp} · Bashir Alam</p>
</div>

<div class="container">

  <!-- STATS -->
  <div class="stat-grid" style="margin: 1.5rem 0;">
    <div class="stat-card">
      <div class="number">{total_models}</div>
      <div class="label">Models Evaluated</div>
    </div>
    <div class="stat-card" style="background: linear-gradient(135deg,#11998e,#38ef7d);">
      <div class="number">{total_scenarios}</div>
      <div class="label">Scenarios Run</div>
    </div>
    <div class="stat-card" style="background: linear-gradient(135deg,#e96c75,#f4a261);">
      <div class="number">{total_samples}</div>
      <div class="label">Total Samples</div>
    </div>
    <div class="stat-card" style="background: linear-gradient(135deg,#4facfe,#00f2fe);">
      <div class="number">{len(all_metrics)}</div>
      <div class="label">Metrics Computed</div>
    </div>
  </div>

  <!-- LEADERBOARD -->
  {self._build_leaderboard(all_results)}

  <!-- CHARTS -->
  <div class="grid-2">
    {plotly_charts}
  </div>

  <!-- PER-SCENARIO TABS -->
  {self._build_scenario_tabs(all_results)}

  <!-- PREDICTIONS BROWSER -->
  {self._build_predictions_browser(all_results)}

</div>

<footer>
  VLM Eval Framework · <a href="https://github.com/bashiralam185/vlm-eval-framework">GitHub</a> ·
  Built by <strong>Bashir Alam</strong> · Åbo Akademi University / AiLiveSim
</footer>

{REPORT_JS}
</body>
</html>
"""
        return body

    def _build_leaderboard(self, all_results: Dict[str, List[ScenarioResult]]) -> str:
        """Build the ranked leaderboard table."""
        rows_data = []
        for model_name, scenario_results in all_results.items():
            for sr in scenario_results:
                agg = sr.aggregate_scores()
                rows_data.append({
                    "model": model_name,
                    "scenario": sr.scenario_name,
                    "task": sr.task_type,
                    "samples": sr.n_samples,
                    "failed": sr.n_failed,
                    **agg,
                })

        # Sort by clip_score if present, else semantic_similarity
        sort_key = "clip_score" if any("clip_score" in r for r in rows_data) else "semantic_similarity"
        rows_data.sort(key=lambda r: r.get(sort_key, 0), reverse=True)

        metric_cols = [k for k in (rows_data[0].keys() if rows_data else [])
                       if k not in ("model", "scenario", "task", "samples", "failed")]

        header_cells = "".join(
            f"<th>{col.replace('_', ' ').title()}</th>" for col in metric_cols
        )

        def score_cell(val):
            bar = f"""<div class="score-bar-bg"><div class="score-bar"
                style="width:{min(val*100,100):.0f}%"></div></div>""" if val <= 1.0 else ""
            return f'<td class="metric">{val:.3f} {bar}</td>'

        table_rows = ""
        for r in rows_data:
            task_class = r["task"].split("_")[0]
            table_rows += f"""
            <tr>
              <td><strong>{r["model"]}</strong></td>
              <td>{r["scenario"]}</td>
              <td><span class="tag {task_class}">{r["task"]}</span></td>
              <td>{r["samples"]}</td>
              {"".join(score_cell(r.get(m, 0.0)) for m in metric_cols)}
            </tr>"""

        return f"""
        <div class="card">
          <div class="card-header">🏆 Leaderboard</div>
          <div class="card-body" style="overflow-x:auto;">
            <table>
              <thead><tr>
                <th>Model</th><th>Scenario</th><th>Task</th><th>Samples</th>
                {header_cells}
              </tr></thead>
              <tbody>{table_rows}</tbody>
            </table>
          </div>
        </div>"""

    def _build_plotly_charts(self, all_results: Dict[str, List[ScenarioResult]]) -> str:
        """Build Plotly bar chart and radar chart as inline divs."""
        try:
            import plotly.graph_objects as go
            import plotly.io as pio

            # Chart 1: Bar chart — average metric per model
            model_names = list(all_results.keys())
            primary_metric = "clip_score"
            bar_vals = []
            for m_name in model_names:
                scores = [
                    sr.aggregate_scores().get(primary_metric, 0.0)
                    for sr in all_results[m_name]
                ]
                bar_vals.append(sum(scores) / len(scores) if scores else 0.0)

            fig_bar = go.Figure(go.Bar(
                x=model_names,
                y=bar_vals,
                marker_color=["#4299e1", "#ed8936", "#48bb78", "#9f7aea", "#f56565"],
                text=[f"{v:.3f}" for v in bar_vals],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title=f"Average {primary_metric.replace('_',' ').title()} by Model",
                yaxis_title="Score",
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(t=50, b=30, l=30, r=30),
                height=300,
            )
            bar_html = pio.to_html(fig_bar, full_html=False, include_plotlyjs=False)

            # Chart 2: Grouped bar by scenario
            scenario_names = list({sr.scenario_name for results in all_results.values() for sr in results})
            fig_grouped = go.Figure()
            colors = ["#4299e1", "#ed8936", "#48bb78", "#9f7aea", "#f56565"]
            for i, m_name in enumerate(model_names):
                y_vals = []
                for sc_name in scenario_names:
                    match = next(
                        (sr for sr in all_results[m_name] if sr.scenario_name == sc_name), None
                    )
                    y_vals.append(match.aggregate_scores().get("semantic_similarity", 0.0) if match else 0.0)
                fig_grouped.add_trace(go.Bar(
                    name=m_name, x=scenario_names, y=y_vals,
                    marker_color=colors[i % len(colors)],
                ))
            fig_grouped.update_layout(
                barmode="group",
                title="Semantic Similarity by Scenario",
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=50, b=30, l=30, r=30),
                height=300,
                legend=dict(orientation="h", y=-0.3),
            )
            grouped_html = pio.to_html(fig_grouped, full_html=False, include_plotlyjs=False)

            return f"""
            <div class="card">
              <div class="card-header">📊 Score by Model</div>
              <div class="card-body">{bar_html}</div>
            </div>
            <div class="card">
              <div class="card-header">📊 Score by Scenario</div>
              <div class="card-body">{grouped_html}</div>
            </div>"""

        except Exception as e:
            logger.warning(f"Could not generate Plotly charts: {e}")
            return ""

    def _build_scenario_tabs(self, all_results: Dict[str, List[ScenarioResult]]) -> str:
        """Build a tabbed view for each scenario."""
        all_scenarios = list({
            sr.scenario_name
            for results in all_results.values()
            for sr in results
        })

        tab_buttons = "".join(
            f'<button class="tab-btn {"active" if i == 0 else ""}" '
            f'data-group="scenarios" data-tab="{sc}" '
            f'onclick="switchTab(\'scenarios\',\'{sc}\')">{sc}</button>'
            for i, sc in enumerate(all_scenarios)
        )

        tab_contents = ""
        for i, sc_name in enumerate(all_scenarios):
            model_rows = ""
            for m_name, results in all_results.items():
                match = next((sr for sr in results if sr.scenario_name == sc_name), None)
                if match is None:
                    continue
                agg = match.aggregate_scores()
                score_cells = "".join(
                    f'<td class="metric">{v:.4f}</td>' for v in agg.values()
                )
                model_rows += f"<tr><td><strong>{m_name}</strong></td>{score_cells}</tr>"

            # Get metric names from first matching result
            first_match = next(
                (sr for results in all_results.values() for sr in results if sr.scenario_name == sc_name),
                None,
            )
            metric_headers = (
                "".join(f"<th>{k.replace('_',' ').title()}</th>" for k in first_match.aggregate_scores().keys())
                if first_match else ""
            )

            tab_contents += f"""
            <div class="tab-content {"active" if i == 0 else ""}"
                 data-group="scenarios" data-tab="{sc_name}">
              <table>
                <thead><tr><th>Model</th>{metric_headers}</tr></thead>
                <tbody>{model_rows}</tbody>
              </table>
            </div>"""

        return f"""
        <div class="card">
          <div class="card-header">🗂 Per-Scenario Results</div>
          <div class="card-body">
            <div class="tabs">{tab_buttons}</div>
            {tab_contents}
          </div>
        </div>"""

    def _build_predictions_browser(self, all_results: Dict[str, List[ScenarioResult]]) -> str:
        """Show sample predictions side-by-side for each model."""
        # Take up to 5 samples from the first scenario
        first_results = next(iter(all_results.values()), [])
        if not first_results:
            return ""

        first_scenario = first_results[0]
        sample_ids = [r.sample_id for r in first_scenario.sample_results[:5]]

        if not sample_ids:
            return ""

        rows = ""
        for sid in sample_ids:
            row_cells = f"<td><code>{sid}</code></td>"
            for m_name, results in all_results.items():
                match_sr = next((sr for sr in results if sr.scenario_name == first_scenario.scenario_name), None)
                if match_sr is None:
                    row_cells += "<td>N/A</td>"
                    continue
                match_r = next((r for r in match_sr.sample_results if r.sample_id == sid), None)
                if match_r is None:
                    row_cells += "<td>N/A</td>"
                    continue
                pred = match_r.prediction or "<em>(failed)</em>"
                score_str = ", ".join(f"{k}: {v:.3f}" for k, v in list(match_r.scores.items())[:3])
                row_cells += f"""
                <td>
                  <div class="pred-box">{pred[:200]}{"..." if len(pred) > 200 else ""}</div>
                  <small style="color:#718096">{score_str}</small>
                </td>"""
            rows += f"<tr>{row_cells}</tr>"

        model_headers = "".join(f"<th>{m}</th>" for m in all_results.keys())

        return f"""
        <div class="card">
          <div class="card-header">🔍 Prediction Browser (Sample: {first_scenario.scenario_name})</div>
          <div class="card-body" style="overflow-x:auto;">
            <table>
              <thead><tr><th>Sample ID</th>{model_headers}</tr></thead>
              <tbody>{rows}</tbody>
            </table>
          </div>
        </div>"""
