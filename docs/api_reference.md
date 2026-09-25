# nn-toolbox API Reference

Complete programmatic reference for functions, classes, and utilities in `nn-toolbox`.

---

## 1. Master Diagnostic Function

### `nn_toolbox.diagnose`
```python
def diagnose(
    model: torch.nn.Module,
    dataloader: Optional[Iterable[Any]] = None,
    loss_fn: Optional[Callable[..., torch.Tensor]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    mode: str = "light",  # "light" | "deep"
    diagnostics: Optional[List[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
    num_batches: int = 2,
    sample_input: Optional[Any] = None,
    sample_target: Optional[Any] = None,
    verbose: bool = True,
) -> DiagnosticReport
```
* **Parameters**:
  * `model`: PyTorch neural network module.
  * `dataloader`: Training DataLoader yielding batches (dict, tuple, or list).
  * `loss_fn`: Loss callable `(outputs, targets) -> Tensor` or `(outputs) -> Tensor`.
  * `optimizer`: Active PyTorch optimizer instance.
  * `mode`: `"light"` for non-destructive passive diagnostics; `"deep"` to include active experiments.
  * `diagnostics`: List of individual diagnostics to enable/disable (e.g. `["overfit", "lr_sweep"]`).
  * `device`: Torch device to execute on (auto-inferred if `None`).
  * `sample_input`: Explicit input tensor override.
  * `sample_target`: Explicit target tensor override.
  * `verbose`: When `True`, formats and prints terminal report.
* **Returns**: `DiagnosticReport` containing findings, metrics, and prioritized investigation targets.

---

## 2. Core Data Models

### `nn_toolbox.core.DiagnosticFinding`
Represents an empirical observation, causal hypotheses, and quantitative evidence.
* **Attributes**:
  * `category`: `str` (e.g. `"forward"`, `"backward"`, `"optimization"`, `"data"`, `"memorization"`).
  * `severity`: `str` (`"info"`, `"warning"`, `"critical"`).
  * `observation`: `str` — empirical quantitative finding.
  * `interpretation`: `str` — cautious contextual explanation.
  * `module`: `Optional[str]` — name of offending module or parameter.
  * `evidence`: `Dict[str, Any]` — raw measurements and statistics.
  * `hypotheses`: `List[str]` — plausible candidate explanations.
  * `suggested_actions`: `List[str]` — actionable steps to investigate or fix.

### `nn_toolbox.core.DiagnosticReport`
Container aggregating diagnostic session telemetry and findings.
* **Properties**:
  * `findings`: `List[DiagnosticFinding]` — all recorded findings.
  * `actionable_findings`: `List[DiagnosticFinding]` — warning and critical findings requiring user action.
  * `healthy_findings`: `List[DiagnosticFinding]` — verified healthy diagnostic dimensions.
  * `critical_findings`: `List[DiagnosticFinding]` — critical severity findings.
  * `warning_findings`: `List[DiagnosticFinding]` — warning severity findings.
  * `metrics`: `Dict[str, Any]` — full quantitative telemetry dictionary.
* **Methods**:
  * `get_investigation_targets() -> List[Dict[str, Any]]`: Computes severity-weighted module ranking.
  * `save_html(filepath: str) -> None`: Renders and saves standalone styled HTML report.
  * `save_json(filepath: str) -> None`: Serializes report to JSON file.
  * `print_summary(verbose: bool = False) -> None`: Displays formatted terminal report.

---

## 3. Instrumentation Monitors

### `nn_toolbox.instrumentation.ActivationMonitor`
```python
ActivationMonitor(
    model: torch.nn.Module,
    sample_limit: int = 4096,
    compute_higher_moments: bool = False,
)
```
* **Context Manager**: `with act_mon:` registers forward hooks, calculates Welford streaming stats online, and cleans up hooks on exit.
* **Methods**:
  * `get_latest_stats() -> Dict[str, Dict[str, Any]]`: Returns per-module statistics.
  * `analyze_signal_propagation(step: int = 0) -> Dict[str, Any]`: Computes layer-to-layer scaling ratios, amplification events, and attenuation events.

### `nn_toolbox.instrumentation.GradientMonitor`
```python
GradientMonitor(
    model: torch.nn.Module,
    sample_limit: int = 4096,
)
```
* **Methods**:
  * `collect_parameter_gradients() -> Dict[str, Dict[str, Any]]`: Inspects `model.named_parameters()` post-backward, recording gradient norm, mean, variance, sparsity, and NaN/Inf presence.
  * `analyze_backward_propagation() -> Dict[str, Any]`: Evaluates gradient reachability, vanishing gradients, exploding gradients, and step cosine similarity.

### `nn_toolbox.instrumentation.ParameterMonitor`
```python
ParameterMonitor(model: torch.nn.Module)
```
* **Methods**:
  * `inspect_parameters() -> Dict[str, Any]`: Returns parameter counts, trainable fractions, and norms.
  * `snapshot_before_step() -> None`: Takes pre-optimizer step parameter clone snapshot.
  * `snapshot_after_step() -> Dict[str, Any]`: Computes displacement $\|\theta_{t+1} - \theta_t\|$ and update-to-weight ratios $\|\Delta\theta\| / \|\theta\|$.

---

## 4. Diagnostic Experiments

### `nn_toolbox.experiments.overfit_test`
```python
overfit_test(
    model: torch.nn.Module,
    dataset_or_batch: Any,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    sizes: Optional[List[int]] = None,      # Default: [1, 2, 8]
    max_steps: int = 80,
    target_loss: float = 0.01,
    learning_rate: float = 1e-3,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]
```
Repeatedly trains on tiny subsets of samples to verify whether the architecture and optimization setup can memorize trivial amounts of data. Restores original weights upon completion.

### `nn_toolbox.experiments.lr_sweep`
```python
lr_sweep(
    model: torch.nn.Module,
    sample_input: Any,
    sample_target: Optional[Any],
    loss_fn: Callable[..., torch.Tensor],
    lr_range: Optional[List[float]] = None, # Default: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    steps_per_lr: int = 15,
) -> Dict[str, Any]
```
Probes learning rates across logarithmic decades and characterizes behavior: `too_little_movement`, `useful_learning`, `unstable`, `divergent`, or `nan`.

### `nn_toolbox.experiments.initialization_diagnostic`
```python
initialization_diagnostic(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    num_samples: int = 2,
) -> Dict[str, Any]
```
Feeds synthetic standard normal inputs $x \sim \mathcal{N}(0, 1)$ into the model to probe initial forward/backward signal scaling in isolation from dataset confounders.

### `nn_toolbox.experiments.train_eval_test`
```python
train_eval_test(
    model: torch.nn.Module,
    sample_input: Any,
    atol: float = 1e-4,
    rtol: float = 1e-2,
) -> Dict[str, Any]
```
Executes forward passes in `model.train()` and `model.eval()` on identical inputs, measuring relative difference and verifying deterministic eval repeatability.

### `nn_toolbox.experiments.gradient_check`
```python
gradient_check(
    module: torch.nn.Module,
    sample_input: torch.Tensor,
    parameters: Optional[List[torch.nn.Parameter]] = None,
    eps: float = 1e-5,
    rtol: float = 1e-2,
) -> Dict[str, Any]
```
Performs central finite-difference numerical gradient checks against autograd gradients, computing relative error $E_{rel}$.

### `nn_toolbox.experiments.perturbation_test`
```python
perturbation_test(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    epsilons: Optional[List[float]] = None, # Default: [1e-4, 1e-3, 1e-2]
) -> Dict[str, Any]
```
Measures directional Lipschitz gain $\|f(x+\epsilon) - f(x)\| / \|\epsilon\|$ to identify hyper-sensitive or locally flat mappings.

### `nn_toolbox.experiments.verify_bootstrapping`
```python
verify_bootstrapping(
    model: torch.nn.Module,
    sample_batch_or_loader: Optional[Any] = None,
    loss_fn: Optional[Callable[..., torch.Tensor]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    bootstrap_epochs: int = 3,
    freeze_param_names: Optional[List[str]] = None,
    target_score: Optional[float] = None,
    min_loss_drop: float = 0.10,
    bootstrap_results: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]
```
Verifies whether Bootstrapping v1.0 kickstarting works effectively. Evaluates parameter freeze isolation, kickstart convergence, loss reduction on sample subsets, and release readiness.

---

## 5. Anomaly Detectors

### `nn_toolbox.detectors.BootstrapDetector`
Monitors and diagnoses Bootstrapping v1.0 kickstarting metrics.
* **Finding Category**: `FindingCategory.BOOTSTRAP` (`"bootstrap"`).
* **Checks**:
  * Frozen parameter gradient isolation: flags gradient leakage into frozen layers with `Severity.CRITICAL`.
  * Numerical divergence: flags loss explosion or NaNs with `Severity.CRITICAL`.
  * Fitting progress: flags insufficient loss reduction ($<$ `min_loss_drop`) or failure to reach acceptable score with `Severity.WARNING`.
  * Successful kickstart confirmation: emits an `Severity.INFO` confirmation verifying that the model achieved acceptable kickstart fit and is primed for full release.
