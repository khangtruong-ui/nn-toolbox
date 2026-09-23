"""Tests for terminal, JSON, and HTML report formatting."""

import json
import os
import tempfile
from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.core.report_data import DiagnosticReport
from nn_toolbox.report.html import render_html_report
from nn_toolbox.report.json import format_json_report
from nn_toolbox.report.terminal import format_terminal_report


def test_reporting_pipeline():
    report = DiagnosticReport(model_name="UNetBackbone", mode="light")
    report.add_finding(
        DiagnosticFinding(
            category=FindingCategory.FORWARD.value,
            severity=Severity.WARNING.value,
            module="encoder.block.11",
            observation="encoder.block.11 activation RMS is 18.2× median",
            interpretation="Consistent with forward amplification",
            hypotheses=["unscaled residual", "missing LayerNorm"],
            suggested_actions=["Check normalization before block 11"],
        )
    )
    report.add_finding(
        DiagnosticFinding(
            category=FindingCategory.OPTIMIZATION.value,
            severity=Severity.INFO.value,
            observation="Parameters are updating cleanly",
        )
    )

    # 1. Terminal report
    term_text = format_terminal_report(report, verbose=True)
    assert "nn-toolbox diagnostic report: UNetBackbone" in term_text
    assert "FORWARD" in term_text
    assert "POSSIBLE INVESTIGATION TARGETS" in term_text
    assert "encoder.block.11" in term_text

    # 2. JSON report
    json_str = format_json_report(report)
    parsed = json.loads(json_str)
    assert parsed["model_name"] == "UNetBackbone"
    assert len(parsed["findings"]) == 2
    assert len(parsed["investigation_targets"]) >= 1

    # 3. HTML report
    html_text = render_html_report(report)
    assert "<!DOCTYPE html>" in html_text
    assert "encoder.block.11" in html_text
    assert "UNetBackbone" in html_text

    # 4. Save to disk
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "report.json")
        html_path = os.path.join(tmpdir, "report.html")
        report.save_json(json_path)
        report.save_html(html_path)
        assert os.path.exists(json_path)
        assert os.path.exists(html_path)
