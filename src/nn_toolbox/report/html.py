"""
HTML report rendering with interactive styling, metric badges, and layer progression tables.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from nn_toolbox.core.report_data import DiagnosticReport


def render_html_report(report: DiagnosticReport) -> str:
    """Render a standalone, styled HTML diagnostic report."""
    summary = report.summary()
    targets = report.get_investigation_targets()

    findings_rows = []
    for f in report.findings:
        sev = f.severity.lower()
        badge_cls = "badge-danger" if sev == "critical" else ("badge-warning" if sev == "warning" else "badge-success")
        mod_label = html.escape(f.module or "Global")
        obs_text = html.escape(f.observation)
        hypo_text = html.escape(", ".join(f.hypotheses)) if f.hypotheses else "None"
        act_text = html.escape(", ".join(f.suggested_actions)) if f.suggested_actions else "None"

        findings_rows.append(f"""
        <tr>
            <td><span class="badge {badge_cls}">{sev.upper()}</span></td>
            <td><strong>{html.escape(f.category.upper())}</strong></td>
            <td><code>{mod_label}</code></td>
            <td>{obs_text}</td>
            <td><small>{hypo_text}</small></td>
            <td><small>{act_text}</small></td>
        </tr>
        """)

    targets_items = []
    for idx, t in enumerate(targets[:8], 1):
        sev = t["max_severity"].lower()
        badge_cls = "badge-danger" if sev == "critical" else "badge-warning"
        hypos = html.escape(", ".join(t["hypotheses"])) if t["hypotheses"] else "Review telemetry"
        actions = html.escape("; ".join(t["suggested_actions"])) if t["suggested_actions"] else "Inspect module"
        targets_items.append(f"""
        <li class="target-item">
            <span class="badge {badge_cls}">{idx}. {html.escape(t['target'])}</span>
            <div class="target-meta">
                <strong>Hypotheses:</strong> {hypos}<br>
                <strong>Suggested Action:</strong> {actions}
            </div>
        </li>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>nn-toolbox Diagnostic Report: {html.escape(report.model_name)}</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #334155;
            --accent-success: #10b981;
            --accent-warning: #f59e0b;
            --accent-danger: #ef4444;
            --accent-info: #3b82f6;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        .badge-danger {{ background: #7f1d1d; color: #fecaca; }}
        .badge-warning {{ background: #78350f; color: #fef3c7; }}
        .badge-success {{ background: #064e3b; color: #a7f3d0; }}
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
        }}
        .card-val {{
            font-size: 28px;
            font-weight: bold;
            margin-top: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            margin-bottom: 24px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
            font-size: 13px;
        }}
        th {{
            background: #111827;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        code {{
            background: #020617;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            color: #38bdf8;
        }}
        .target-list {{
            list-style: none;
            padding: 0;
            margin: 0 0 24px 0;
        }}
        .target-item {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 10px;
        }}
        .target-meta {{
            margin-top: 6px;
            font-size: 13px;
            color: var(--text-secondary);
        }}
        .disclaimer {{
            font-size: 12px;
            color: var(--text-secondary);
            font-style: italic;
            border-top: 1px solid var(--border-color);
            padding-top: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 nn-toolbox Diagnostic Report</h1>
            <div>Model: <strong>{html.escape(report.model_name)}</strong> | Mode: <strong>{html.escape(report.mode.upper())}</strong></div>
        </header>

        <div class="cards-grid">
            <div class="card">
                <div>CRITICAL FINDINGS</div>
                <div class="card-val" style="color: var(--accent-danger)">{summary['critical_count']}</div>
            </div>
            <div class="card">
                <div>WARNING FINDINGS</div>
                <div class="card-val" style="color: var(--accent-warning)">{summary['warning_count']}</div>
            </div>
            <div class="card">
                <div>PASS / INFORMATIONAL</div>
                <div class="card-val" style="color: var(--accent-success)">{summary['info_count']}</div>
            </div>
            <div class="card">
                <div>INVESTIGATION TARGETS</div>
                <div class="card-val" style="color: var(--accent-info)">{len(targets)}</div>
            </div>
        </div>

        <h2>🎯 Top Investigation Targets</h2>
        <ul class="target-list">
            {"".join(targets_items) if targets_items else "<li class='target-item'>No anomalous investigation targets identified.</li>"}
        </ul>

        <h2>📋 Observations & Hypotheses</h2>
        <table>
            <thead>
                <tr>
                    <th>Severity</th>
                    <th>Category</th>
                    <th>Module</th>
                    <th>Observation</th>
                    <th>Hypotheses</th>
                    <th>Suggested Action</th>
                </tr>
            </thead>
            <tbody>
                {"".join(findings_rows)}
            </tbody>
        </table>

        <div class="disclaimer">
            Note: Diagnostic findings represent empirical observations and hypotheses based on quantitative measurements. They do not constitute guaranteed defects.
        </div>
    </div>
</body>
</html>
"""
    return html_content
