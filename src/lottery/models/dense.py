import torch
from torch import nn


class LeNet300100(nn.Module):
    """The 784-300-100-10 fully-connected network used for MNIST in the LTH paper."""

    def __init__(self, num_classes: int = 10, in_features: int = 28 * 28) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)  # type: ignore[no-any-return]
