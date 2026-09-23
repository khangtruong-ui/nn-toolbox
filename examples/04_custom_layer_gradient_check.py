"""
Example 4: "My custom layer may have an incorrect gradient."

Scenario:
A developer implements a custom autograd backward function (e.g. specialized attention,
kernel, or loss). Training is erratic or diverges.
nn-toolbox runs targeted finite-difference numerical gradient checks on specific parameters:
- Compares analytical autograd gradients vs empirical finite difference slopes
- Measures relative error ||g_num - g_auto|| / (||g_num|| + ||g_auto||)
- Pinpoints which parameter formulation has a faulty backward pass
"""

import torch
import torch.nn as nn
from nn_toolbox.experiments.gradient_check import gradient_check


class CustomAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        ctx.save_for_backward(q, k)
        return torch.matmul(q, k.t())

    @staticmethod
    def backward(ctx, grad_output):
        q, k = ctx.saved_tensors
        # Correct gradient for q: grad_out @ k
        grad_q = torch.matmul(grad_output, k)
        # BUG: intentionally wrong backward formula for k (e.g. missing transpose or wrong variable)
        grad_k = torch.matmul(grad_output.t(), q) * 0.1  # Wrong scale factor!
        return grad_q, grad_k


class CustomLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_q = nn.Parameter(torch.randn(4, 4))
        self.w_k = nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        q = torch.matmul(x, self.w_q)
        k = torch.matmul(x, self.w_k)
        return CustomAttnFunction.apply(q, k)


def main():
    print("--- Example 4: Numerical Gradient Check on Custom Layer ---")

    layer = CustomLayer()
    x = torch.randn(2, 4)

    print("\nRunning gradient_check on ['w_q', 'w_k']:")
    res = gradient_check(layer, x, parameters=["w_q", "w_k"])

    print(f"\nOverall Passed: {res['all_passed']}")
    for param_name, result in res["param_results"].items():
        print(f"  Parameter: {param_name}")
        print(f"    Status: {result['status']}")
        print(f"    Max Relative Error: {result['max_relative_error']:.2e}")

    print("\nDiagnostic Findings:")
    for f in res["findings"]:
        print(f"[{f.severity.upper()}] {f.observation}")
        print(f"  Interpretation: {f.interpretation}")
        print(f"  Hypotheses: {f.hypotheses}")


if __name__ == "__main__":
    main()
