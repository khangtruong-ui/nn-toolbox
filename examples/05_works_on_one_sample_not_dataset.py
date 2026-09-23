"""
Example 5: "My model works on one sample but not on the real dataset."

Scenario:
The model fits a single sample or tiny batch easily, but fails completely on the real dataset.
nn-toolbox runs automated overfitting tests across multiple dataset sizes [1, 2, 8, 32]
along with a learning rate sweep to determine if the issue is:
- Optimization scale / learning rate mismatch with larger batches
- Model capacity saturation
- Data normalization or diversity
"""

import torch
import torch.nn as nn
from nn_toolbox.experiments.lr_sweep import lr_sweep
from nn_toolbox.experiments.overfit import overfit_test


class BottleneckModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Tiny capacity bottleneck: 64 -> 2 -> 64 (cannot represent diverse data)
        self.encoder = nn.Linear(64, 2)
        self.decoder = nn.Linear(2, 64)

    def forward(self, x):
        return self.decoder(torch.relu(self.encoder(x)))


def main():
    print("--- Example 5: One-Sample Overfitting vs Dataset Capacity ---")

    model = BottleneckModel()
    x = torch.randn(32, 64)
    y = x.clone()  # Autoencoder reconstruction task
    loss_fn = nn.MSELoss()

    print("\nRunning overfit_test on sizes [1, 2, 8, 32]:")
    res = overfit_test(model, {"x": x, "y": y}, loss_fn=loss_fn, sizes=[1, 2, 8, 32], max_steps=80)

    for r in res["results"]:
        status_str = "MEMORIZED" if r["memorized"] else "FAILED TO MEMORIZE"
        print(f"  Tiny Dataset ({r['size']} samples): Final Loss {r['final_loss']:.4f} -> {status_str}")

    print("\nDiagnostic Findings:")
    for f in res["findings"]:
        print(f"[{f.severity.upper()}] {f.observation}")
        print(f"  Interpretation: {f.interpretation}")
        print(f"  Hypotheses: {f.hypotheses}")


if __name__ == "__main__":
    main()
