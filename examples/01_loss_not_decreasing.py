"""
Example 1: "My loss does not decrease."

Scenario:
The user notices that training loss remains flat across epochs.
nn-toolbox runs targeted diagnostics to isolate the cause:
- Checks if parameters have requires_grad=True
- Checks if gradients are non-zero
- Checks parameter update-to-weight ratio ||Delta theta|| / ||theta||
- Runs a 1-sample and 8-sample overfit test
"""

import torch
import torch.nn as nn
from nn_toolbox import diagnose


def main():
    print("--- Example 1: Diagnosing Stagnant Training Loss ---")

    # Intentionally broken setup: learning rate is 0.0, so weights never update
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # Bug: zero LR
    loss_fn = nn.MSELoss()

    x = torch.randn(16, 10)
    y = torch.randn(16, 1)

    print("\nRunning diagnose(..., mode='light'):")
    report = diagnose(
        model=model,
        sample_input=x,
        sample_target=y,
        loss_fn=loss_fn,
        optimizer=optimizer,
        mode="light",
        verbose=True,
    )

    print("\nKey Takeaways from Report:")
    for target in report.get_investigation_targets():
        print(f"Target: {target['target']}")
        print(f"  Hypotheses: {target['hypotheses']}")
        print(f"  Suggested Action: {target['suggested_actions']}")


if __name__ == "__main__":
    main()
