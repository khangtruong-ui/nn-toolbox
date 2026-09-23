# nn-toolbox

> **Systematic diagnostic and investigation laboratory for neural-network training.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)

The goal of `nn-toolbox` is **not merely to monitor metrics**. It executes lightweight measurements and controlled experiments that help distinguish between possible root causes of training failure.

### Central Philosophy

> **Observe → formulate failure hypotheses → run targeted diagnostics → collect evidence → report likely investigation targets.**

Diagnostics produce **evidence**, not definitive claims:
* **Observation**: *"Block 11 activation RMS is 14.7× the model median."*
* **Interpretation**: *"This is unusually large relative to neighboring blocks."*
* **Possible Hypotheses**:
  - Activation amplification / unscaled residual connection
  - Normalization layer placement or missing epsilon
  - Unusual input distribution
  - Intentional architectural behavior

---

## Installation

```bash
pip install -e .
```

Or with optional dependencies:
```bash
pip install -e ".[all]"
```

---

## Quickstart

Run a full non-destructive diagnostic on your model, dataloader, and optimizer with a single command:

```python
import torch
from nn_toolbox import diagnose

# Your standard PyTorch setup
model = MyModel()
dataloader = MyDataLoader()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Run diagnostic session
report = diagnose(
    model=model,
    dataloader=dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    mode="light",  # "light" (fast passive checks) or "deep" (includes active experiments)
    verbose=True,
)

# Export results
report.save_json("diagnostic_report.json")
report.save_html("diagnostic_report.html")
```

### Sample Terminal Output

```text
============================================================
 nn-toolbox diagnostic report: UNet (Mode: LIGHT)
============================================================

FORWARD
  ✓ activation statistics collected across 18 layers
  ⚠ [encoder.block.11] activation RMS is 18.2× median

BACKWARD
  ✓ no global gradient explosion
  ⚠ [decoder.block.7] gradient RMS is unusually large (24.1× median)

OPTIMIZATION
  ✓ parameters are updating
  ⚠ [decoder.block.7] update/weight ratio is high (0.42)

TRAIN/EVAL
  ✓ no suspicious discrepancy (relative diff: 0.00%)

DATA
  ✓ inputs and targets finite, normalized in range [0.0, 1.0]

POSSIBLE INVESTIGATION TARGETS
  1. [WARNING] decoder.block.7 - localized gradient amplification, high update ratio
  2. [WARNING] encoder.block.11 - unscaled residual, missing LayerNorm

These are hypotheses based on observed measurements,
not confirmed causes.
============================================================
```

---

## Core Capabilities & Experiments

### 1. Passive Instrumentation (Zero Memory Overhead)
- **Streaming statistics**: Calculates `mean`, `std`, `RMS`, `L1/L2 norm`, percentiles, zero fraction, and NaN/Inf counts online without retaining large activation tensors.
- **Forward & Backward Signal Propagation**: Relative layer-to-layer scaling ratios ($std_l / std_{l-1}$) and gradient norm progression across depth.
- **Parameter & Update Tracking**: Distinguishes raw gradient magnitude from actual parameter update displacement:
  $$\frac{\|\Delta \theta\|}{\|\theta\|}$$

### 2. Targeted Diagnostic Experiments
Run any experiment independently:

```python
from nn_toolbox.experiments import (
    overfit_test,
    lr_sweep,
    initialization_diagnostic,
    train_eval_test,
    gradient_check,
    perturbation_test,
    ablation_test,
    label_shuffle_test,
)
```

| Experiment | Function | What It Answers |
|---|---|---|
| **Memorization Test** | `overfit_test(model, data, loss_fn, sizes=[1, 2, 8])` | Can the model memorize a tiny dataset? Separates pipeline failure from capacity/data issues. |
| **Learning Rate Sweep** | `lr_sweep(model, x, y, loss_fn, lr_range=[...])` | Characterizes LR response: stagnation, useful learning, instability, divergence, or NaNs. |
| **Initialization Diagnostic** | `initialization_diagnostic(model, input_shape)` | Feeds synthetic $x \sim \mathcal{N}(0, 1)$ to isolate init/architecture bugs from training dynamics. |
| **Train/Eval Consistency** | `train_eval_test(model, x)` | Compares `model.train()` vs `model.eval()`. Isolates Dropout, BatchNorm, and mode-dependent shifts. |
| **Numerical Gradient Check** | `gradient_check(module, x, parameters=[...])` | Finite-difference vs autograd verification for custom layers and loss formulations. |
| **Perturbation Sensitivity** | `perturbation_test(model, x, epsilons=[...])` | Measures empirical Lipschitz gain $\|f(x+\epsilon) - f(x)\| / \|\epsilon\|$ without constructing Jacobians. |
| **Representation Collapse** | `analyze_representation_collapse(features)` | Checks pairwise cosine similarity and effective rank to detect dimensional collapse. |
| **Label Shuffle Sanity** | `label_shuffle_test(model, x, y, loss_fn)` | Compares learning descent on true labels vs randomized labels. |

---

## 7 Common Debugging Recipes

Runnable examples are provided in the `examples/` directory:

1. **"My loss does not decrease."** (`examples/01_loss_not_decreasing.py`)
   Checks requires_grad, gradient existence, optimizer update-to-weight ratios, and runs 1-sample overfit.
2. **"My gradients explode."** (`examples/02_exploding_gradients.py`)
   Traces activation scale amplification across depth and isolates the offending layer.
3. **"My model trains but validation is terrible."** (`examples/03_validation_is_terrible.py`)
   Tests train/eval mode shifts and runs label shuffle capacity analysis.
4. **"My custom layer may have an incorrect gradient."** (`examples/04_custom_layer_gradient_check.py`)
   Runs finite-difference numerical gradient validation on custom autograd functions.
5. **"My model works on one sample but not on the real dataset."** (`examples/05_works_on_one_sample_not_dataset.py`)
   Runs multi-size overfit tests [1, 2, 8, 32] to distinguish capacity bottlenecks from optimization scale.
6. **"My Transformer representation seems collapsed."** (`examples/06_representation_collapse.py`)
   Computes effective rank, pairwise similarity, and attention distribution entropy.
7. **"My model behaves differently between train() and eval()."** (`examples/07_train_eval_discrepancy.py`)
   Isolates running-statistics distribution shifts and stochastic layers.

---

## Architecture-Aware Analyzers

* **CNN Analyzer** (`CNNAnalyzer`): Checks spatial dimension progression, receptive fields, and premature spatial downsampling collapse.
* **Transformer Analyzer** (`TransformerAnalyzer`): Checks attention entropy, token sinks, and query-key dot-product scaling.

---

## License

Licensed under the [Apache License 2.0](LICENSE).
