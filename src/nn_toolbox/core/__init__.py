"""Core definitions and finding representations for nn-toolbox."""

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.core.report_data import DiagnosticReport
from nn_toolbox.core.tensor_utils import (
    DEFAULT_PRIMARY_OUTPUT_KEYS,
    extract_output_tensors,
    extract_primary_tensor,
)

__all__ = [
    "DiagnosticFinding",
    "FindingCategory",
    "Severity",
    "DiagnosticReport",
    "DEFAULT_PRIMARY_OUTPUT_KEYS",
    "extract_output_tensors",
    "extract_primary_tensor",
]
