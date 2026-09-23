"""
Terminal reporting with structured sections, clear observations, and investigation targets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from nn_toolbox.core.report_data import DiagnosticReport


def format_terminal_report(report: DiagnosticReport, verbose: bool = False) -> str:
    """Format DiagnosticReport into an elegant, concise terminal representation."""
    lines: List[str] = []
    bar = "=" * 60

    lines.append(bar)
    lines.append(f" nn-toolbox diagnostic report: {report.model_name} (Mode: {report.mode.upper()})")
    summary = report.summary()
    healthy_cnt = summary.get("healthy_count", len(report.healthy_findings))
    warn_cnt = summary.get("warning_count", len(report.warning_findings))
    crit_cnt = summary.get("critical_count", len(report.critical_findings))
    lines.append(f" Status: {healthy_cnt} verified healthy | {warn_cnt} warning(s) | {crit_cnt} critical")
    lines.append(bar)

    sections = [
        ("FORWARD", "forward"),
        ("BACKWARD", "backward"),
        ("OPTIMIZATION", "optimization"),
        ("MEMORIZATION", "memorization"),
        ("TRAIN/EVAL", "train_eval"),
        ("DATA", "data"),
        ("STABILITY", "stability"),
        ("INITIALIZATION", "initialization"),
    ]

    for section_title, category_key in sections:
        cat_findings = report.get_findings_by_category(category_key)
        if not cat_findings:
            continue

        lines.append("")
        lines.append(f"{section_title}")

        for f in cat_findings:
            if f.severity == "critical":
                symbol = "✗"
            elif f.severity == "warning":
                symbol = "⚠"
            else:
                symbol = "✓"

            prefix = f"  {symbol}"
            mod_prefix = f" [{f.module}]" if f.module else ""
            lines.append(f"{prefix}{mod_prefix} {f.observation}")

            if verbose and f.interpretation:
                lines.append(f"     Interpretation: {f.interpretation}")
            if verbose and f.hypotheses:
                lines.append(f"     Hypotheses: {', '.join(f.hypotheses)}")

    # Targets
    targets = report.get_investigation_targets()
    lines.append("")
    lines.append("POSSIBLE INVESTIGATION TARGETS")
    if not targets:
        lines.append("  No urgent anomalies detected.")
    else:
        for idx, t in enumerate(targets[:6], 1):
            target_name = t["target"]
            sev = t["max_severity"].upper()
            hypo_str = f" - {', '.join(t['hypotheses'][:2])}" if t["hypotheses"] else ""
            lines.append(f"  {idx}. [{sev}] {target_name}{hypo_str}")

    lines.append("")
    lines.append("These are hypotheses based on observed measurements,")
    lines.append("not confirmed causes.")
    lines.append(bar)

    return "\n".join(lines)
