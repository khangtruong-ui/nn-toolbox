"""
Example 2: "My gradients explode."

Scenario:
During training, loss suddenly shoots to Inf or NaN.
nn-toolbox isolates the exact layer amplifying signals and causing gradient explosion:
- Identifies which layer's gradient norm exceeds the median by >20x
- Verifies forward activation variance across depth
- Formulates hypotheses (unscaled residual addition, missing LayerNorm)
"""

import torch
import torch.nn as nn
from nn_toolbox import diagnose


class UnscaledDeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Intentionally unscaled large weights causing explosion
        self.l1 = nn.Linear(16, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 64)
        self.l4 = nn.Linear(64, 2)
        with torch.no_grad():
            self.l1.weight.mul_(3.0)
            self.l2.weight.mul_(3.0)
            self.l3.weight.mul_(3.0)

    def forward(self, x):
        return self.l4(torch.relu(self.l3(torch.relu(self.l2(torch.relu(self.l1(x)))))))


def main():
    print("--- Example 2: Diagnosing Gradient and Activation Explosion ---")

    model = UnscaledDeepNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    x = torch.randn(8, 16)
    y = torch.randn(8, 2)

    report = diagnose(
        model=model,
        sample_input=x,
        sample_target=y,
        loss_fn=loss_fn,
        optimizer=optimizer,
        mode="light",
        verbose=True,
    )

    print("\nActionable Findings:")
    for f in report.actionable_findings:
        print(f"[{f.severity.upper()}] {f.category}: {f.observation}")
        print(f"  Interpretation: {f.interpretation}")
        print(f"  Hypotheses: {f.hypotheses}")


if __name__ == "__main__":
    main()
