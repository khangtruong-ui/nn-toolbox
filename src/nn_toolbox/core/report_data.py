"""
Report data structure holding diagnostic results, metrics, findings, and targets.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nn_toolbox.core.finding import DiagnosticFinding, Severity


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic container aggregating findings, measurements,

    and prioritized investigation targets.
    """

    model_name: str = "Model"
    timestamp: float = field(default_factory=time.time)
    mode: str = "light"
    findings: List[DiagnosticFinding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_finding(self, finding: DiagnosticFinding) -> None:
        """Append a finding to the report."""
        self.findings.append(finding)

    def add_findings(self, findings: List[DiagnosticFinding]) -> None:
        """Append multiple findings."""
        self.findings.extend(findings)

    @property
    def actionable_findings(self) -> List[DiagnosticFinding]:
        """Return findings that require attention (warning or critical)."""
        return [f for f in self.findings if f.is_actionable()]

    @property
    def critical_findings(self) -> List[DiagnosticFinding]:
        """Return findings with critical severity."""
        return [f for f in self.findings if f.severity in (Severity.CRITICAL.value, "critical")]

    @property
    def warning_findings(self) -> List[DiagnosticFinding]:
        """Return findings with warning severity."""
        return [f for f in self.findings if f.severity in (Severity.WARNING.value, "warning")]

    @property
    def healthy_findings(self) -> List[DiagnosticFinding]:
        """Return findings representing verified healthy or passing diagnostic checks."""
        return [f for f in self.findings if not f.is_actionable()]

    def get_findings_by_category(self, category: str) -> List[DiagnosticFinding]:
        """Filter findings by category."""
        cat_lower = category.lower()
        return [f for f in self.findings if f.category.lower() == cat_lower]

    def get_investigation_targets(self) -> List[Dict[str, Any]]:
        """Rank modules and architectural components by finding severity and count."""
        target_scores: Dict[str, Dict[str, Any]] = {}

        severity_weights = {
            Severity.CRITICAL.value: 10,
            Severity.WARNING.value: 3,
            Severity.INFO.value: 1,
        }

        for finding in self.findings:
            if not finding.is_actionable():
                continue

            target_key = finding.module or finding.category
            if not target_key:
                target_key = "global"

            is_frozen = bool(
                finding.evidence.get("is_frozen", False)
                if isinstance(finding.evidence, dict)
                else False
            ) or "[FROZEN]" in finding.observation

            if target_key not in target_scores:
                target_scores[target_key] = {
                    "target": target_key,
                    "module": finding.module,
                    "category": finding.category,
                    "score": 0,
                    "findings_count": 0,
                    "max_severity": finding.severity,
                    "is_frozen": is_frozen,
                    "hypotheses": set(),
                    "suggested_actions": set(),
                }

            weight = severity_weights.get(finding.severity, 1)
            target_scores[target_key]["score"] += weight
            target_scores[target_key]["findings_count"] += 1
            if is_frozen:
                target_scores[target_key]["is_frozen"] = True
            if weight > severity_weights.get(target_scores[target_key]["max_severity"], 0):
                target_scores[target_key]["max_severity"] = finding.severity

            for h in finding.hypotheses:
                target_scores[target_key]["hypotheses"].add(h)
            for a in finding.suggested_actions:
                target_scores[target_key]["suggested_actions"].add(a)

        sorted_targets = sorted(
            target_scores.values(),
            key=lambda x: (
                0 if x.get("is_frozen", False) else 1,
                x["score"],
                x["findings_count"],
            ),
            reverse=True,
        )

        # Convert sets to lists
        result = []
        for item in sorted_targets:
            result.append({
                "target": item["target"],
                "module": item["module"],
                "category": item["category"],
                "score": item["score"],
                "findings_count": item["findings_count"],
                "max_severity": item["max_severity"],
                "is_frozen": item.get("is_frozen", False),
                "hypotheses": sorted(list(item["hypotheses"])),
                "suggested_actions": sorted(list(item["suggested_actions"])),
            })
        return result

    def summary(self) -> Dict[str, Any]:
        """Return a high-level summary of report metrics and findings."""
        return {
            "model_name": self.model_name,
            "mode": self.mode,
            "total_findings": len(self.findings),
            "critical_count": len(self.critical_findings),
            "warning_count": len(self.warning_findings),
            "healthy_count": len(self.healthy_findings),
            "info_count": len(self.healthy_findings),
            "investigation_targets_count": len(self.get_investigation_targets()),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full dictionary representation."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "summary": self.summary(),
            "findings": [f.to_dict() for f in self.findings],
            "investigation_targets": self.get_investigation_targets(),
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """JSON serialized report."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, filepath: str) -> None:
        """Write JSON report to disk."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def save_html(self, filepath: str) -> None:
        """Render and save HTML report."""
        from nn_toolbox.report.html import render_html_report
        import os
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        html_content = render_html_report(self)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

    def print_summary(self, verbose: bool = False) -> None:
        """Print terminal diagnostic report."""
        from nn_toolbox.report.terminal import format_terminal_report
        print(format_terminal_report(self, verbose=verbose))
