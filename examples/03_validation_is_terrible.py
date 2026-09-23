"""
Example 3: "My model trains but validation is terrible."

Scenario:
The training loss descends, but validation performance fails or diverges.
nn-toolbox runs train vs eval consistency and label sensitivity experiments:
- Compares model outputs under model.train() vs model.eval()
- Checks if Dropout / BatchNorm or custom running stats are causing an unexpected distribution shift
- Evaluates model memorization capacity vs generalization via label shuffle test
"""

import torch
import torch.nn as nn
from nn_toolbox.experiments.label_shuffle import label_shuffle_test
from nn_toolbox.experiments.train_eval import train_eval_test


class ModelWithAggressiveDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 128)
        self.drop1 = nn.Dropout(p=0.8)  # Extremely high dropout
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(p=0.8)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        h1 = self.drop1(torch.relu(self.fc1(x)))
        h2 = self.drop2(torch.relu(self.fc2(h1)))
        return self.fc3(h2)


def main():
    print("--- Example 3: Train vs Eval Discrepancy & Validation Failure ---")

    model = ModelWithAggressiveDropout()
    x = torch.randn(16, 32)
    y = torch.randn(16, 2)
    loss_fn = nn.MSELoss()

    print("\n1. Running train/eval consistency test:")
    te_results = train_eval_test(model, x)
    print(f"Relative output shift between train() and eval(): {te_results['relative_difference']*100:.1f}%")
    for f in te_results.get("findings", []):
        print(f"  [{f.severity.upper()}] {f.observation}")
        print(f"    Interpretation: {f.interpretation}")

    print("\n2. Running label shuffle test to check capacity vs generalization:")
    ls_results = label_shuffle_test(model, x, y, loss_fn=loss_fn, steps=20)
    print(f"True labels loss reduction: {ls_results['true_drop']*100:.1f}%")
    print(f"Shuffled labels loss reduction: {ls_results['shuffled_drop']*100:.1f}%")
    for f in ls_results.get("findings", []):
        print(f"  [{f.severity.upper()}] {f.observation}")


if __name__ == "__main__":
    main()
