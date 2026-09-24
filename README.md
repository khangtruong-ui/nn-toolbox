# nn-toolbox: Neural Network Diagnostic & Investigation Laboratory

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![Tests: Passing](https://img.shields.io/badge/Tests-35%2F35%20passing-brightgreen.svg)](tests/)

An automated diagnostic and investigation framework for PyTorch neural network training.

The goal of `nn-toolbox` is **not merely to monitor metrics**. It performs targeted measurements and controlled micro-experiments that systematically isolate and identify the underlying causes of training failure.

---

## Central Philosophy

> **Observe → Formulate failure hypotheses → Run targeted diagnostics → Collect evidence → Report likely investigation targets.**

Most ML debugging tools act like passive dashboards: they plot a loss curve that diverges or stalls, leaving the engineer to guess why. `nn-toolbox` acts like an automated **diagnostic laboratory**. When a symptom is observed, it formulates hypotheses, executes targeted measurements, collects quantitative evidence, and prioritizes actionable investigation targets.

### Evidence-Based, Not Claim-Based
`nn-toolbox` deliberately produces **evidence and causal hypotheses**, not dogmatic assertions:
* **Prefer**: *"Block 11 activation variance is 18.2× larger than the preceding block (std: 14.7 vs 0.81). This pattern is consistent with possible forward-signal amplification or an unscaled residual connection."*
* **Rather than**: *"Block 11 is broken."*

### Verified Healthy Confirmations
When layers, signal pathways, or optimization dynamics are functioning normally, `nn-toolbox` **concisely reports verified healthy dimensions**. Knowing with certainty that gradients flow to 100% of trainable parameters and that update-to-weight ratios are within the optimal regime allows practitioners to rule out entire classes of failure immediately.

---

## Table of Contents

- [Core Capabilities](#core-capabilities)
- [Mathematical & Diagnostic Foundations](#mathematical--diagnostic-foundations)
  - [1. Update-to-Weight Ratio (||\Delta\theta|| / ||\theta||)](#1-update-to-weight-ratio--delta-theta----theta-)
  - [2. Streaming Statistics via Welford's Algorithm](#2-streaming-statistics-via-welfords-algorithm)
  - [3. Signal Propagation & Layer-Relative Scaling](#3-signal-propagation--layer-relative-scaling)
  - [4. Representation Collapse & Effective Subspace Rank](#4-representation-collapse--effective-subspace-rank)
  - [5. Local Lipschitz Sensitivity](#5-local-lipschitz-sensitivity)
  - [6. Numerical Gradient Verification](#6-numerical-gradient-verification)
  - [7. Inter-Branch Gradient Scale Balance](#7-inter-branch-gradient-scale-balance)
  - [8. Graph Connectivity & Disconnected Subgraph Analysis](#8-graph-connectivity--disconnected-subgraph-analysis)
  - [9. Evaluation Mode Determinism & Stochastic Drift](#9-evaluation-mode-determinism--stochastic-drift)
- [Package Architecture](#package-architecture)
- [Installation](#installation)
- [Quickstart: Programmatic Usage](#quickstart-programmatic-usage)
  - [1. One-Line Master Diagnostic (`diagnose`)](#1-one-line-master-diagnostic-diagnose)
  - [2. Passive Instrumentation Monitors](#2-passive-instrumentation-monitors)
  - [3. Active Diagnostic Experiments](#3-active-diagnostic-experiments)
- [Diagnostic Experiments Suite](#diagnostic-experiments-suite)
- [Anomaly Detectors Suite](#anomaly-detectors-suite)
- [Verified Healthy Confirmations](#verified-healthy-confirmations)
- [Prioritized Investigation Targets](#prioritized-investigation-targets)
- [Reporting: Terminal, JSON & HTML](#reporting-terminal-json--html)
- [CLI Reference (`nn-diagnose`)](#cli-reference-nn-diagnose)
- [Integration with Existing Pipelines (SID-UNet Example)](#integration-with-existing-pipelines-sid-unet-example)
- [7 Common Debugging Recipes](#7-common-debugging-recipes)
- [Documentation & Guides](#documentation--guides)
- [License](#license)

---

## Core Capabilities

* **Passive Instrumentation**:
  * Online streaming statistics (mean, std, RMS, L1/L2 norm, percentiles, min/max, zero/NaN/inf fractions, skewness, kurtosis) without storing full activation tensors.
  * Layer-relative signal propagation tracking ($std_l / std_{l-1}$).
  * True parameter update tracking ($\|\Delta \theta\| / \|\theta\|$).
  * Gradient reachability, norm progression, and step cosine similarity.
  * Tensor memory layout and contiguity analysis (detects non-contiguous slices and stride mismatches).
  * Inter-branch gradient balance tracking across architectural components.
* **Targeted Experiments**:
  * **Automated Memorization / Overfitting Test**: Probes model and optimization capacity on $N \in \{1, 2, 8, 32\}$ samples.
  * **Logarithmic Learning Rate Sweep**: Evaluates training dynamics from $10^{-6}$ to $10^{-1}$ to identify optimal, stagnant, oscillatory, and divergent regimes.
  * **Initialization Diagnostic**: Isolates architecture-level signal flow from dataset confounders using synthetic $x \sim \mathcal{N}(0, 1)$.
  * **Train/Eval Consistency Diagnostic**: Compares mode-dependent forward behavior to detect running-statistics shifts or unseeded stochastic operations.
  * **Evaluation Mode Determinism Test**: Verifies exact bitwise and relative output reproducibility across repeated evaluation passes.
  * **Numerical Gradient Checking**: Finite-difference verification for custom autograd functions and loss formulations.
  * **Perturbation / Local Lipschitz Sensitivity**: Probes directional feature sensitivity $\|\Delta y\| / \|\Delta x\|$.
  * **Representation Collapse Analysis**: Computes effective rank via SVD entropy to detect subspace collapse.
  * **Label Shuffle Sanity**: Tests optimization against randomized targets to establish data signal-to-noise bounds.
* **Architecture-Aware Analyzers & Graph Detectors**:
  * **Graph Connectivity Detector**: Identifies and clusters trainable parameters structurally disconnected from loss backward pass.
  * **Gradient Balance Detector**: Pinpoints branch dominance and gradient starvation in multi-head and composite networks.
  * **Tensor Layout Detector**: Flags non-contiguous intermediate slices that trigger `.view()` crashes.
  * **CNN Analyzer**: Detects premature spatial downsampling, bottleneck collapse, and channel redundancy.
  * **Transformer Analyzer**: Measures multi-head attention entropy, attention collapse, and token sinks.
* **Multi-Format Reporting**:
  * Interactive standalone HTML reports with metric badges and layer progression tables.
  * Clean ANSI terminal tables with symbol-coded findings (`✓`, `⚠`, `✗`).
  * Machine-readable structured JSON outputs.

---

## Mathematical & Diagnostic Foundations

### 1. Update-to-Weight Ratio ($||\Delta\theta|| / ||\theta||$)

A common fallacy in deep learning debugging is assuming that large gradient norms mean large weight updates, or that small gradient norms mean training is stalled. In modern adaptive optimizers (AdamW, RMSprop, Adafactor), gradient magnitude is normalized by second-moment accumulators:

$$\Delta \theta_t = -\eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t$$

`nn-toolbox` monitors the true empirical parameter displacement between optimizer steps:

$$u(\theta) = \frac{\|\theta_{t+1} - \theta_t\|_2}{\|\theta_t\|_2 + \epsilon}$$

#### The 4 Optimization Regimes
| Regime | Metric Range | Diagnosis & Interpretation |
|---|---|---|
| **Stagnant** | $u(\theta) < 10^{-6}$ | Updates are near or below the floating-point noise floor. Training is effectively stalled. Causes: learning rate too small, frozen weights, or saturated zero gradients. |
| **Weak / Slow** | $10^{-6} \le u(\theta) < 10^{-4}$ | Learning is occurring but at an excessively sluggish pace. Often requires 5–10× learning rate boost. |
| **Effective** | $10^{-4} \le u(\theta) \le 10^{-2}$ | **Healthy optimal regime**. Parameters make steady, stable progress through the loss landscape. |
| **Divergent** | $u(\theta) > 0.1$ | Step displacement is excessively large relative to weight magnitude. Causes: learning rate too high, gradient explosion, or missing weight decay. |

---

### 2. Streaming Statistics via Welford's Algorithm

Storing full intermediate activations across tens or hundreds of layers quickly exhausts GPU VRAM, causing out-of-memory crashes during diagnosis.

`nn-toolbox` implements **Welford's single-pass online algorithm** to compute accurate mean, variance, and higher moments in a single pass without caching intermediate activation tensors:

$$M_1 = x_1, \quad S_1 = 0$$
$$M_k = M_{k-1} + \frac{x_k - M_{k-1}}{k}$$
$$S_k = S_{k-1} + (x_k - M_{k-1})(x_k - M_k)$$
$$\sigma^2 = \frac{S_n}{n - 1}$$

This guarantees numerical stability even when activation values have large offsets, with **zero persistent GPU memory overhead**.

---

### 3. Signal Propagation & Layer-Relative Scaling

Arbitrary universal thresholds (e.g. *"activation std must be between 0.5 and 1.5"*) fail across diverse architectures. Instead, `nn-toolbox` evaluates **layer-to-layer relative scaling ratios**:

$$R_{fwd}(l) = \frac{\sigma(h_l)}{\sigma(h_{l-1}) + \epsilon}$$

* **Forward Signal Amplification**: $R_{fwd}(l) > 2.0$ sustained across multiple layers, or $R_{fwd}(l) > 20.0$ in a single layer (critical explosion).
* **Forward Signal Attenuation**: $R_{fwd}(l) < 0.2$ sustained, or $R_{fwd}(l) < 0.05$ (signal collapse / dead ReLUs).
* **Stable Propagation**: $0.5 \le R_{fwd}(l) \le 1.8$ across all network layers.

An analogous check tracks gradient propagation during backpropagation:

$$R_{bwd}(l) = \frac{\|\nabla_{\theta_l} L\|_2}{\text{median}_{k}(\|\nabla_{\theta_k} L\|_2) + \epsilon}$$

---

### 4. Representation Collapse & Effective Subspace Rank

When intermediate representations collapse into a low-dimensional subspace, downstream layers receive redundant or rank-deficient features.

Given a batch activation matrix $H \in \mathbb{R}^{B \times D}$ with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_K > 0$, `nn-toolbox` normalizes the singular value spectrum into a probability distribution:

$$p_i = \frac{\sigma_i}{\sum_{j=1}^K \sigma_j}$$

The **Effective Rank** is defined via spectral Shannon entropy:

$$H(p) = -\sum_{i=1}^K p_i \ln p_i, \quad R_{eff} = \exp(H(p))$$

* If $R_{eff} \approx 1$, all representations are aligned along a single 1D trajectory (**complete dimensional collapse**).
* If $R_{eff} \approx \min(B, D)$, representations span diverse orthogonal subspaces (**healthy full-rank representation**).

---

### 5. Local Lipschitz Sensitivity

The local Lipschitz sensitivity measures how aggressively perturbations in input or intermediate activations scale at the output:

$$S(x, \epsilon) = \frac{\|f(x + \epsilon) - f(x)\|_2}{\|\epsilon\|_2}$$

* **Hyper-sensitivity ($S > 100$)**: The model is ill-conditioned; minor input variations or numerical rounding cause wild output swings.
* **Insensitivity ($S < 10^{-5}$)**: The function is locally flat, indicating dead activation regions or zero gradients.
* **Well-conditioned ($0.1 \le S \le 10$)**: Stable, robust functional mapping.

---

### 6. Numerical Gradient Verification

For custom layers or loss functions, `nn-toolbox` performs two-sided central finite-difference verification:

$$\nabla_{num} f(\theta)_i = \frac{f(\theta + h e_i) - f(\theta - h e_i)}{2h}$$

Relative error between autograd $\nabla_{auto}$ and numerical $\nabla_{num}$:

$$E_{rel} = \frac{\|\nabla_{auto} - \nabla_{num}\|_2}{\|\nabla_{auto}\|_2 + \|\nabla_{num}\|_2 + \epsilon}$$

* $E_{rel} < 10^{-5}$: **PASS** (exact autograd gradient).
* $10^{-5} \le E_{rel} < 10^{-2}$: **ACCEPTABLE** (minor floating point discrepancy).
* $E_{rel} \ge 10^{-2}$: **FAIL** (analytical gradient formula bug).

---

### 7. Inter-Branch Gradient Scale Balance

In multi-branch architectures (e.g., latent diffusion models with parallel encoders/decoders or multi-task networks with auxiliary classifier heads), gradient magnitudes across branches can diverge by several orders of magnitude. 

Comparing raw L2 parameter norms across different layers is flawed because L2 norm naturally scales with $\sqrt{N}$ (a 560,000-parameter convolution filter has an L2 norm $\sim 130\times$ larger than a 32-element bias even with identical per-element gradient distributions). `nn-toolbox` computes the scale-invariant Root-Mean-Square (RMS) gradient per parameter and aggregates medians across architectural branches:

$$RMS(\nabla_{\theta_b}) = \sqrt{\frac{1}{N_b} \sum_{i=1}^{N_b} (\nabla_{\theta_{b,i}})^2}$$

The inter-branch disparity ratio is evaluated:

$$\mathcal{D} = \frac{\max_{b} \text{median}(RMS(\nabla_{\theta_b}))}{\min_{b} \text{median}(RMS(\nabla_{\theta_b}))}$$

* $\mathcal{D} < 50$: **Balanced**: Gradients distribute smoothly across sub-networks.
* $50 \le \mathcal{D} \le 500$: **Mild Disparity**: Normal in specialized projection layers or sparse attention blocks.
* $\mathcal{D} > 500$: **Severe Starvation / Dominance**: The dominant branch monopolizes learning while the starved branch receives near-zero effective updates. Indicates unbalanced multi-task loss weighting or severe gradient attenuation.

---

### 8. Graph Connectivity & Disconnected Subgraph Analysis

Parameters marked `requires_grad=True` that receive zero gradients across consecutive optimization steps waste GPU memory, allocate unused optimizer states (Adam first/second moments), and frequently indicate hidden computational graph detachments:
1. Submodules instantiated as trainable but omitted from `model.forward()`.
2. Intermediate tensors detached from autograd via `.detach()`, `.item()`, or conversions to NumPy/Python scalars.
3. Auxiliary loss components whose loss weight is zero or whose outputs are omitted from the objective.

`GraphConnectivityDetector` aggregates zero-gradient parameters by submodule prefix (e.g., `vae.decoder`), identifying whether an entire computational branch is dead rather than just individual un-activated biases.

---

### 9. Evaluation Mode Determinism & Stochastic Drift

Inference pipelines, validation loops, and diagnostic benchmarks depend on deterministic, reproducible execution. Uncontrolled stochasticity during `eval()`—such as unseeded Gaussian noise injection, active dropout, or running statistics mutation—creates metric jitter, noisy validation curves, and nondeterministic inference bugs.

`eval_determinism_test` executes multiple repeated forward passes on identical batches under `model.eval()`, measuring the maximum absolute difference and relative stochastic drift:

$$\Delta_{rel} = \frac{\max_{k > 1} \|f_k(x) - f_1(x)\|_2}{\|f_1(x)\|_2 + \epsilon}$$

* $\Delta_{rel} < 10^{-4}$: **Deterministic**: Identical, bitwise-consistent evaluation outputs.
* $\Delta_{rel} \ge 10^{-4}$: **Non-Deterministic**: Flagged with diagnostic findings identifying stochastic layers or non-seeded random perturbations.

---

## Package Architecture

```text
nn-toolbox/
├── src/nn_toolbox/
│   ├── __init__.py                # Package exports & versioning
│   ├── diagnose.py                # Master high-level orchestrator & CLI entrypoint
│   │
│   ├── core/                      # Data models & report structures
│   │   ├── finding.py             # DiagnosticFinding, Severity, FindingCategory
│   │   └── report_data.py         # DiagnosticReport, metrics aggregation, target ranking
│   │
│   ├── instrumentation/           # Zero-overhead hooks & telemetry
│   │   ├── hooks.py               # HookManager (safe forward & backward lifecycle)
│   │   ├── activations.py         # ActivationMonitor (Welford streaming stats)
│   │   ├── gradients.py           # GradientMonitor (norms, reachability, cosine similarity)
│   │   ├── parameters.py          # ParameterMonitor (||Δθ|| / ||θ|| tracking)
│   │   └── storage.py             # RollingMetricsStorage (windowed temporal history)
│   │
│   ├── experiments/               # Targeted active diagnostic experiments
│   │   ├── overfit.py             # Automated tiny-dataset memorization test
│   │   ├── lr_sweep.py            # Logarithmic learning rate sensitivity sweep
│   │   ├── initialization.py      # Synthetic normal forward/backward probe
│   │   ├── train_eval.py          # Train vs Eval consistency test
│   │   ├── eval_determinism.py    # Evaluation mode determinism & drift diagnostic
│   │   ├── gradient_check.py      # Finite-difference autograd numerical check
│   │   ├── perturbation.py        # Local Lipschitz sensitivity probe
│   │   ├── ablation.py            # Zeroing & layer importance ablation
│   │   └── label_shuffle.py       # Random-target signal-to-noise sanity
│   │
│   ├── detectors/                 # Rule-based failure hypothesis detectors
│   │   ├── base.py                # BaseDetector interface
│   │   ├── connectivity.py        # Disconnected subgraphs & unhooked parameters
│   │   ├── gradient_balance.py    # Inter-branch gradient scale balance & starvation
│   │   ├── layout.py              # Non-contiguous tensor memory layout & slicing
│   │   ├── exploding.py           # Exploding activations & gradients detector
│   │   ├── vanishing.py           # Vanishing signals & dead gradients detector
│   │   ├── saturation.py          # Activation saturation detector (zero/max fraction)
│   │   ├── collapse.py            # Representation rank collapse detector
│   │   ├── instability.py         # Loss & gradient oscillation detector
│   │   ├── optimization.py        # Stagnant updates & frozen weights detector
│   │   └── data.py                # Constant inputs, NaNs, Infs, bad variance detector
│   │
│   ├── analysis/                  # Mathematical & statistical utilities
│   │   ├── statistics.py          # Welford accumulation & tensor summary
│   │   ├── distributions.py       # Percentile estimation & histogram generation
│   │   ├── similarity.py          # SVD effective rank & cosine similarity
│   │   └── sensitivity.py         # Lipschitz ratio calculations
│   │
│   ├── analyzers/                 # Architecture-specialized structural analyzers
│   │   ├── cnn.py                 # CNNAnalyzer (spatial collapse, channel ratios)
│   │   └── transformer.py         # TransformerAnalyzer (attention entropy, token sinks)
│   │
│   └── report/                    # Reporting backends
│       ├── terminal.py            # ANSI colored terminal formatter
│       ├── json.py                # Structured JSON serializer
│       └── html.py                # Standalone responsive HTML generator
│
├── tests/                         # Comprehensive unit & pathology test suite (35 tests)
│   ├── test_architectures_and_pathologies.py
│   ├── test_detectors.py
│   ├── test_experiments.py
│   ├── test_instrumentation.py
│   ├── test_reporting.py
│   └── test_new_learnability_suites.py
├── examples/                      # 7 runnable debugging recipe scripts
└── docs/                          # In-depth technical guides
```

---

## Installation

Install in editable mode in your active Python environment:

```bash
cd /workspace/nn-toolbox
pip install -e .
```

To install with full optional dependencies (testing, plotting, formatting):
```bash
pip install -e ".[all]"
```

---

## Quickstart: Programmatic Usage

### 1. One-Line Master Diagnostic (`diagnose`)

```python
import torch
import torch.nn as nn
from nn_toolbox import diagnose

# Define or load your PyTorch model, dataloader, loss function, and optimizer
model = MyUNet()
dataloader = MyDataLoader()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Run automated diagnostic session
report = diagnose(
    model=model,
    dataloader=dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    mode="light",   # "light" for fast passive checks; "deep" for active experiments
    verbose=True,   # Prints elegant terminal summary
)

# Export reports
report.save_html("diagnostic_report.html")
report.save_json("diagnostic_report.json")

# Inspect programmatically
print(f"Verified healthy dimensions: {len(report.healthy_findings)}")
print(f"Actionable issues found: {len(report.actionable_findings)}")
for target in report.get_investigation_targets():
    print(f"Target: {target['target']} | Hypotheses: {target['hypotheses']}")
```

### 2. Passive Instrumentation Monitors

Monitors can be attached directly to your existing training loop with zero code modification:

```python
from nn_toolbox.instrumentation import ActivationMonitor, GradientMonitor, ParameterMonitor

act_mon = ActivationMonitor(model)
grad_mon = GradientMonitor(model)
param_mon = ParameterMonitor(model)

with act_mon, grad_mon:
    # Forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    
    # Backward pass
    loss.backward()
    
    # Check gradient flow across parameters
    grad_mon.collect_parameter_gradients()
    bwd_analysis = grad_mon.analyze_backward_propagation()
    
    # Measure true parameter displacement before and after optimizer step
    param_mon.snapshot_before_step()
    optimizer.step()
    updates = param_mon.snapshot_after_step()
    
    print(f"Update-to-weight ratio: {updates['global_update_ratio']:.2e}")
```

### 3. Active Diagnostic Experiments

Run individual experiments to investigate specific failure hypotheses:

```python
from nn_toolbox.experiments import overfit_test, lr_sweep, train_eval_test

# 1. Can the model memorize 1, 2, or 8 samples?
of_results = overfit_test(model, dataloader, loss_fn=loss_fn, sizes=[1, 8])
print(f"Memorized 1 sample: {of_results['results'][0]['memorized']}")

# 2. How does the loss respond to learning rates from 1e-6 to 1e-1?
lr_results = lr_sweep(model, sample_input, sample_target, loss_fn=loss_fn)
print(f"Suggested LR regime: {lr_results['suggested_lr']}")

# 3. Are train() and eval() modes behaving consistently?
te_results = train_eval_test(model, sample_input)
print(f"Train/eval relative difference: {te_results['relative_difference']:.2%}")
```

---

## Diagnostic Experiments Suite

| Experiment | Function | Target Pathology |
|---|---|---|
| **Tiny-Batch Memorization** | `overfit_test(...)` | Distinguishes optimization/gradient failure from model capacity constraints. |
| **Learning Rate Sweep** | `lr_sweep(...)` | Characterizes LR response regimes: stagnation, progress, oscillation, divergence. |
| **Initialization Probe** | `initialization_diagnostic(...)` | Evaluates pure architectural signal scaling under $x \sim \mathcal{N}(0, 1)$. |
| **Train/Eval Consistency** | `train_eval_test(...)` | Detects improper BatchNorm running statistics or mode discrepancies. |
| **Eval Determinism Test** | `eval_determinism_test(...)` | Detects uncontrolled stochasticity, unseeded noise, or non-deterministic layers in `eval()`. |
| **Numerical Gradient Check** | `gradient_check(...)` | Finite-difference autograd check for custom autograd functions. |
| **Perturbation Sensitivity** | `perturbation_test(...)` | Measures empirical Lipschitz gain $\|f(x+\epsilon) - f(x)\| / \|\epsilon\|$. |
| **Layer Ablation Test** | `ablation_test(...)` | Evaluates layer importance and zeroing sensitivity. |
| **Label Shuffle Sanity** | `label_shuffle_test(...)` | Compares true label descent against randomized targets to test signal floor. |

---

## Anomaly Detectors Suite

| Detector | Category | Signals & Anomalies Detected |
|---|---|---|
| `GraphConnectivityDetector` | Architecture | Trainable submodules or clusters of parameters disconnected from backward loss computation (0 gradients). |
| `GradientBalanceDetector` | Backward | Inter-branch gradient scale imbalance and layer starvation/dominance across architectural components ($>500\times$ RMS disparity). |
| `TensorLayoutDetector` | Architecture | Non-contiguous intermediate/output tensor layouts and stride mismatches from slicing/permutations that cause `.view()` failures. |
| `ExplodingDetector` | Forward / Backward | Activation variance explosion ($>20\times$), gradient norms exceeding $10^4$ or per-layer gradient explosion. |
| `VanishingDetector` | Forward / Backward | Activation scale collapse ($<0.05\times$), vanishing gradients ($<10^{-8}$), unhooked parameters. |
| `OptimizationDetector` | Optimization | Stagnant updates ($u(\theta) = 0$), tiny update ratios ($u(\theta) < 10^{-6}$), all parameters frozen. |
| `SaturationDetector` | Activation | Extreme zero fractions ($>95\%$ dead ReLUs) or saturated sigmoid/tanh activations ($>95\%$). |
| `CollapseDetector` | Architecture | Representation dimensional collapse via SVD effective rank ratio ($R_{eff} < 0.1$). |
| `InstabilityDetector` | Optimization | Severe loss oscillation, sign-flipping gradient cosine similarities ($\cos < -0.7$). |
| `DataDetector` | Data | Constant batch inputs (variance $<10^{-12}$), NaNs, Infs, extreme input ranges. |

---

## Verified Healthy Confirmations

When diagnostic checks confirm normal operation, `nn-toolbox` reports concise positive confirmations:

```text
============================================================
 nn-toolbox diagnostic report: UNet (Mode: LIGHT)
 Status: 4 verified healthy | 0 warning(s) | 0 critical
============================================================

FORWARD
  ✓ Forward activation propagation is stable across 16 layer(s) (std range: 0.56 - 1.00).

BACKWARD
  ✓ Active gradient flow verified on 100% of trainable parameters (48/48, mean norm: 3.42e-02).

OPTIMIZATION
  ✓ Healthy parameter update ratio: ||Δθ||/||θ|| = 1.81e-02 (displacement norm: 4.59e-01).

DATA
  ✓ Input data is finite and non-constant (variance: 1.00e+00 range: [-4.37, 4.07]).

POSSIBLE INVESTIGATION TARGETS
  No urgent anomalies detected.
============================================================
```

---

## Prioritized Investigation Targets

When multiple anomalies occur, they often cascade. For example, an exploding activation at layer 3 causes exploding gradients at layer 1 and high update ratios at layer 2.

`nn-toolbox` aggregates findings across all detectors and ranks the underlying modules using weighted severity scoring:

$$\text{Score}(M) = 10 \times N_{\text{critical}} + 3 \times N_{\text{warning}} + 1 \times N_{\text{info}}$$

```text
POSSIBLE INVESTIGATION TARGETS
  1. [CRITICAL] encoder.block.2 - Exploding upstream activations, missing normalization
  2. [WARNING] decoder.conv1.weight - Vanishing backward gradient signal
```

Each target carries specific, ranked **hypotheses** and **suggested actions** to guide remediation.

---

## Reporting: Terminal, JSON & HTML

### 1. Interactive HTML Report
Saved via `report.save_html("report.html")`. Features:
* Clean dark-mode dashboard.
* Summary cards for critical, warning, healthy, and target counts.
* Prioritized investigation targets with ranked hypotheses and suggested remediation actions.
* Comprehensive findings table with severity badges.

### 2. Structured JSON Report
Saved via `report.save_json("report.json")`. Contains full quantitative telemetry, metrics dictionaries, and machine-readable findings for CI/CD integration.

### 3. Terminal Report
Printed automatically via `report.print_summary()` or `diagnose(..., verbose=True)`. Uses standard ANSI symbols (`✓`, `⚠`, `✗`) for clean terminal readability.

---

## CLI Reference (`nn-diagnose`)

The package provides a command-line interface `nn-diagnose`:

```bash
# Check CLI installation
nn-diagnose --help

# Run diagnostic and save reports
nn-diagnose --mode deep --json report.json --html report.html
```

---

## Integration with Existing Pipelines (SID-UNet Example)

`nn-toolbox` integrates seamlessly into existing PyTorch training frameworks. In **SID-UNet** (`ai-detection-unet-gpu`), it is invoked before the main training loop via the `--debug` and `--debug-mode` flags:

```bash
# Light mode: Non-destructive runtime learnability check
sid-train --config configs/default.yaml --debug

# Deep mode: Runs tiny-batch memorization & learning rate sweep
sid-train --config configs/default.yaml --debug --debug-mode deep
```

The trainer logs brief healthy confirmations and flags actionable learnability issues before training commences, saving diagnostic reports to `reports/diagnostics/diagnostic_report.html` and `.json`.

---

## 7 Common Debugging Recipes

Runnable scripts demonstrating how to diagnose classic neural network failure modes are provided in `examples/`:

| Recipe | Script | Description |
|---|---|---|
| 1 | [`examples/01_loss_not_decreasing.py`](examples/01_loss_not_decreasing.py) | Diagnoses frozen parameters, zero learning rate, and dead backward gradients. |
| 2 | [`examples/02_exploding_gradients.py`](examples/02_exploding_gradients.py) | Traces activation amplification across depth to isolate unscaled residual connections. |
| 3 | [`examples/03_validation_is_terrible.py`](examples/03_validation_is_terrible.py) | Detects train/eval normalization distribution shifts and label capacity mismatch. |
| 4 | [`examples/04_custom_layer_gradient_check.py`](examples/04_custom_layer_gradient_check.py) | Uses finite-difference gradient checking to identify broken analytical gradients. |
| 5 | [`examples/05_works_on_one_sample_not_dataset.py`](examples/05_works_on_one_sample_not_dataset.py) | Tests memorization scaling ($N=1, 2, 8, 32$) to separate capacity from optimization failure. |
| 6 | [`examples/06_representation_collapse.py`](examples/06_representation_collapse.py) | Identifies dimensional representation collapse in Transformer multi-head attention. |
| 7 | [`examples/07_train_eval_discrepancy.py`](examples/07_train_eval_discrepancy.py) | Isolates mode-dependent shifts in BatchNorm and Dropout. |

---

## Documentation & Guides

For in-depth guides and API documentation:
* [Learnability Diagnostics Guide](docs/learnability_diagnostics_guide.md): In-depth tutorial on diagnosing unlearnable models.
* [API Reference](docs/api_reference.md): Detailed class, method, and function signatures.

---

## License

`nn-toolbox` is open-source software licensed under the [Apache License 2.0](LICENSE).
