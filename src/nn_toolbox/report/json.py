"""
JSON report generator for machine-readable diagnostics.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from nn_toolbox.core.report_data import DiagnosticReport


def format_json_report(report: DiagnosticReport, indent: int = 2) -> str:
    """Serialize DiagnosticReport into structured JSON string."""
    return json.dumps(report.to_dict(), indent=indent, default=str)
