"""
Example 7: "My model behaves differently between train() and eval()."

Scenario:
The user notices that predictions during training evaluate well, but calling model.eval()
causes performance to drop sharply or change unpredictably.
nn-toolbox isolates the exact modules (Dropout, BatchNorm, custom layers) responsible:
- Measures output divergence ||f_train(x) - f_eval(x)||
- Verifies deterministic evaluation repeatability
- Distinguishes expected stochastic behaviors from buggy state logic
"""

import torch
import torch.nn as nn
from nn_toolbox.experiments.train_eval import train_eval_test


class CustomBatchNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        return self.head(torch.relu(self.bn(self.conv(x))))


def main():
    print("--- Example 7: Investigating Train vs Eval Discrepancies ---")

    model = CustomBatchNormNet()
    # If BatchNorm has not seen multiple batches, its running mean/var are uninitialized (defaults: mean=0, var=1)
    # causing a substantial discrepancy between train mode (batch stats) and eval mode (running stats)
    x = torch.randn(4, 3, 32, 32) + 5.0  # Shifted distribution

    print("\nRunning train_eval_test:")
    res = train_eval_test(model, x)

    print(f"Relative output difference: {res['relative_difference'] * 100:.2f}%")
    print(f"Deterministic in eval mode: {res['is_eval_deterministic']}")
    print(f"Has BatchNorm: {res['has_batchnorm']}")
    print(f"Has Dropout: {res['has_dropout']}")

    print("\nFindings:")
    for f in res["findings"]:
        print(f"[{f.severity.upper()}] {f.observation}")
        print(f"  Interpretation: {f.interpretation}")
        print(f"  Hypotheses: {f.hypotheses}")


if __name__ == "__main__":
    main()
