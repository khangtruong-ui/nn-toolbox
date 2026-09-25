"""
nn-toolbox: Systematic diagnostic and investigation laboratory for neural network training.
"""

from nn_toolbox.core.finding import DiagnosticFinding, FindingCategory, Severity
from nn_toolbox.core.report_data import DiagnosticReport
from nn_toolbox.detectors.bootstrap import BootstrapDetector
from nn_toolbox.diagnose import diagnose
from nn_toolbox.experiments import (
    ablation_test,
    eval_determinism_test,
    gradient_check,
    initialization_diagnostic,
    label_shuffle_test,
    lr_sweep,
    overfit_test,
    perturbation_test,
    train_eval_test,
    verify_bootstrapping,
)
from nn_toolbox.instrumentation import (
    ActivationMonitor,
    GradientMonitor,
    HookManager,
    ParameterMonitor,
)

__version__ = "0.1.0"

__all__ = [
    "diagnose",
    "DiagnosticReport",
    "DiagnosticFinding",
    "FindingCategory",
    "Severity",
    "BootstrapDetector",
    "HookManager",
    "ActivationMonitor",
    "GradientMonitor",
    "ParameterMonitor",
    "overfit_test",
    "lr_sweep",
    "initialization_diagnostic",
    "train_eval_test",
    "eval_determinism_test",
    "gradient_check",
    "perturbation_test",
    "ablation_test",
    "label_shuffle_test",
    "verify_bootstrapping",
    "__version__",
]
