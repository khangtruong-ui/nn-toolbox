# In-Depth Guide: Diagnosing Learnability at Runtime

This guide provides a systematic methodology for diagnosing and resolving the most common neural network training pathologies using `nn-toolbox`.

---

## 1. Pathology: Loss Does Not Decrease

### Diagnostic Checklist
When your loss curve remains flat or hovers around initial loss, `nn-toolbox` investigates five distinct failure modes:

```text
Flat Loss Symptom
 ├── 1. Are weights actually receiving updates? (||Δθ|| / ||θ||)
 ├── 2. Do gradients reach all trainable parameters? (bwd reachability)
 ├── 3. Are all model parameters frozen? (requires_grad check)
 ├── 4. Can the model memorize a single sample? (overfit_test N=1)
 └── 5. Is the learning rate in an effective regime? (lr_sweep)
```

### Targeted Measurements

#### Step A: Check Parameter Update Magnitude
```python
from nn_toolbox.instrumentation import ParameterMonitor

param_mon = ParameterMonitor(model)
param_mon.snapshot_before_step()
optimizer.step()
updates = param_mon.snapshot_after_step()

print("Global update ratio:", updates["global_update_ratio"])
```
* **Diagnosis if `global_update_ratio == 0.0`**: Gradients exist, but parameters did not move.
  - *Hypothesis 1*: Optimizer learning rate is `0.0`.
  - *Hypothesis 2*: Parameter tensors passed to optimizer do not match model parameters.
  - *Hypothesis 3*: `GradScaler` skipped optimizer step due to inf/nan gradients.
* **Diagnosis if `global_update_ratio < 1e-6`**: Stagnant update regime.
  - *Action*: Scale up learning rate by 5× to 10×.

#### Step B: Check Gradient Flow & Detached Graph
```python
from nn_toolbox.instrumentation import GradientMonitor

grad_mon = GradientMonitor(model)
loss.backward()
grad_mon.collect_parameter_gradients()
analysis = grad_mon.analyze_backward_propagation()

print("Zero gradient parameters:", analysis["zero_grad_params"])
print("Vanishing parameters:", analysis["vanishing_params"])
```
* **Diagnosis if `len(zero_grad_params) > 0`**:
  - Trainable parameters are disconnected from the loss graph.
  - Search code for `.detach()`, conversion to NumPy (`.numpy()`, `.item()`), or unused submodules.

#### Step C: Run 1-Sample Memorization Probe
```python
from nn_toolbox.experiments import overfit_test

res = overfit_test(model, (batch_x[:1], batch_y[:1]), loss_fn=loss_fn, sizes=[1], max_steps=50)
print("1-sample memorized:", res["results"][0]["memorized"])
```
* If a model cannot memorize a single sample:
  - The problem is **not** data complexity, capacity, or generalization.
  - The problem is an architectural, gradient flow, or loss formulation defect (e.g. inverted loss sign, incorrect target alignment).

---

## 2. Pathology: Loss Explodes to NaN or Inf

### Diagnostic Checklist
```text
Exploding Loss Symptom
 ├── 1. Where does the forward activation variance explode? (fwd propagation)
 ├── 2. Are input values finite and normalized? (data sanity)
 ├── 3. Which layer generates the first NaN/Inf gradient? (gradient monitor)
 └── 4. Is the learning rate beyond the stability frontier? (lr_sweep)
```

### Targeted Measurements

#### Step A: Identify Activation Scale Amplification
```python
from nn_toolbox.instrumentation import ActivationMonitor

with ActivationMonitor(model) as act_mon:
    _ = model(sample_input)
    analysis = act_mon.analyze_signal_propagation()

for event in analysis["amplification_events"]:
    print(f"Module: {event['module']} | Ratio to previous: {event['ratio_to_prev']:.1f}x")
```
* **Interpretation**:
  - An activation ratio $> 20\times$ between adjacent layers pinpoints the exact module amplifying signal.
  - Common causes: unscaled residual additions ($x + f(x)$ without $1/\sqrt{2}$ or LayerNorm), missing $\epsilon$ in custom division, or exponential activation functions.

#### Step B: Probe Learning Rate Stability Frontier
```python
from nn_toolbox.experiments import lr_sweep

sweep = lr_sweep(model, sample_input, sample_target, loss_fn=loss_fn)
for r in sweep["results"]:
    print(f"LR: {r['lr']:.1e} | Status: {r['behavior']} | Loss: {r['final_loss']}")
```
* **Action**: Identify the highest learning rate that maintains `useful_learning` before `unstable` or `divergent` occurs.

---

## 3. Pathology: Model Works on One Sample, but Fails on Real Dataset

### Diagnostic Checklist
When a model passes a 1-sample test but fails on the full training dataset, the failure is **not** a basic implementation bug.

```text
Dataset Failure Symptom
 ├── 1. Memorization scaling test: N = 1, 2, 8, 32 (overfit_test)
 ├── 2. Representation collapse / rank analysis (analyze_representation_collapse)
 └── 3. Label shuffle comparison (label_shuffle_test)
```

### Targeted Measurements

#### Step A: Multi-Size Memorization Curve
```python
from nn_toolbox.experiments import overfit_test

of_res = overfit_test(model, dataloader, loss_fn=loss_fn, sizes=[1, 2, 8, 32])
for r in of_res["results"]:
    print(f"Size {r['size']:2d}: Initial {r['initial_loss']:.3f} -> Final {r['final_loss']:.4f} (Memorized: {r['memorized']})")
```
* **Pattern Analysis**:
  - `N=1 PASS, N=2 PASS, N=8 FAIL`: Suggests bottleneck capacity or layer width limitation.
  - `N=1 PASS, N=8 PASS, N=32 FAIL`: Optimization capacity or learning rate schedule issue under batching.

#### Step B: Effective Rank of Intermediate Features
```python
from nn_toolbox.analysis.similarity import analyze_representation_collapse

rep = analyze_representation_collapse(intermediate_features)
print(f"Effective Rank: {rep['effective_rank']:.2f} / Max: {rep['max_possible_rank']}")
```
* If effective rank is $< 10\%$ of max possible rank, intermediate features suffer from **dimensional collapse**. Increase latent width, apply dropout, or use contrastive regularization.

---

## 4. Pathology: Train vs Eval Discrepancy

### Diagnostic Checklist
```text
Train/Eval Discrepancy Symptom
 ├── 1. Deterministic eval test: Does eval() produce identical outputs?
 ├── 2. Normalization shift: Does BatchNorm running mean/var diverge from batch stats?
 └── 3. Stochastic leakage: Does Dropout remain active during eval()?
```

### Targeted Measurements

```python
from nn_toolbox.experiments import train_eval_test

te = train_eval_test(model, sample_input)
print("Relative difference:", f"{te['relative_difference']:.2%}")
print("Eval is deterministic:", te["is_eval_deterministic"])
```
* **Interpretation**:
  - If `is_eval_deterministic == False`: An unseeded random operation (Dropout, random sampling) is active during `model.eval()`. Check for `if self.training:` guards in custom layers.
  - If discrepancy is $> 500\%$ without Dropout: BatchNorm running statistics have drifted from current batch statistics, typically caused by training on highly heterogeneous batches or too high `momentum`.

---

## 5. Pathology: Custom Layer Produces Incorrect Gradients

### Targeted Measurements

```python
from nn_toolbox.experiments import gradient_check

chk = gradient_check(my_custom_layer, sample_x, eps=1e-5)
for p_name, res in chk["parameter_results"].items():
    print(f"{p_name}: relative error {res['relative_error']:.2e} -> {res['status']}")
```
* If `relative_error > 1e-2`, the analytical backward method has a mathematical derivation bug or tensor indexing mismatch.

---

## 6. Pathology: Disconnected Submodules & Zero-Gradient Parameters

### Diagnostic Checklist
```text
Zero-Gradient Parameters Symptom
 ├── 1. Were submodules instantiated with requires_grad=True but unused in forward()?
 ├── 2. Was an intermediate tensor detached (.detach(), .item(), or numpy)?
 └── 3. Was an auxiliary loss weight set to 0.0 or omitted from total loss?
```

### Targeted Measurements
```python
from nn_toolbox.detectors.connectivity import GraphConnectivityDetector

detector = GraphConnectivityDetector()
findings = detector.detect(context)
for f in findings:
    print(f"Severity: {f.severity} | Module: {f.module} | Issue: {f.observation}")
```
* **Interpretation**:
  - Parameters marked `requires_grad=True` with no backward loss connection waste memory and optimizer tracking.
  - Grouping zero-gradient parameters hierarchically identifies dead subgraphs (e.g., `vae.decoder`) vs isolated dead neurons.

---

## 7. Pathology: Multi-Branch Gradient Starvation & Dominance

### Diagnostic Checklist
```text
Branch Starvation Symptom
 ├── 1. Do loss scales across composite terms differ by > 100x?
 ├── 2. Does an auxiliary head or decoder receive attenuated gradients?
 └── 3. Is the gradient RMS disparity across branches > 500x?
```

### Targeted Measurements
```python
from nn_toolbox.detectors.gradient_balance import GradientBalanceDetector

detector = GradientBalanceDetector()
findings = detector.detect(context)
for f in findings:
    print(f"Dominant branch: {f.evidence['dominant_branch']} vs Starved: {f.evidence['starved_branch']}")
```
* **Interpretation**:
  - Scale-invariant RMS gradient comparison prevents parameter count bias ($\sqrt{N}$) when comparing conv layers against linear heads.
  - Rebalance composite loss weights (e.g. `loss = loss_mask + 0.1 * loss_aux`) to restore gradient flow to starved branches.

---

## 8. Pathology: Non-Contiguous Memory Layouts & Stride Mismatches

### Diagnostic Checklist
```text
Non-Contiguous Layout Symptom
 ├── 1. Were tensors sliced along spatial or channel dimensions (e.g. [:, :, :H, :W])?
 ├── 2. Were tensors permuted/transposed without .contiguous()?
 └── 3. Did downstream code call .view(...) instead of .reshape(...)?
```

### Targeted Measurements
```python
from nn_toolbox.detectors.layout import TensorLayoutDetector

detector = TensorLayoutDetector()
findings = detector.detect(context)
for f in findings:
    print(f"Module: {f.module} | Layout: {f.evidence}")
```
* **Remediation**:
  - Add `.contiguous()` immediately following spatial cropping or permutations.
  - Replace rigid `.view(...)` calls with flexible `.reshape(...)` in analysis and library modules.
