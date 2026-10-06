from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lottery.training import (
    ClassificationTrainer,
    EpochCallback,
    EpochResult,
    Metrics,
    StepCallback,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


class TinyNet(nn.Module):
    def __init__(self, in_features: int = 8, hidden: int = 16, classes: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.bn(self.fc1(x))))


@dataclass
class ShiftTrainer:
    """Deterministic stand-in trainer: every step adds ``delta`` to all trainable parameters.

    Lets tests reason exactly about what weights look like at any step.
    """

    steps_per_epoch: int = 3
    delta: float = 1.0
    val_accuracy: float | None = None
    fit_calls: list[int] = field(default_factory=list)

    def fit(
        self,
        model: nn.Module,
        epochs: int,
        on_step: StepCallback | None = None,
        on_epoch: EpochCallback | None = None,
    ) -> list[EpochResult]:
        self.fit_calls.append(epochs)
        step = 0
        results = []
        for epoch in range(epochs):
            for _ in range(self.steps_per_epoch):
                with torch.no_grad():
                    for p in model.parameters():
                        p.add_(self.delta)
                step += 1
                if on_step is not None:
                    on_step(step, model)
            m = Metrics(loss=1.0 / (epoch + 1), accuracy=0.5)
            val = None if self.val_accuracy is None else Metrics(0.0, self.val_accuracy)
            results.append(EpochResult(epoch=epoch, train=m, test=m, val=val))
            if on_epoch is not None:
                on_epoch(results[-1])
        return results

    def evaluate(self, model: nn.Module, device: torch.device | None = None) -> Metrics:
        return Metrics(loss=0.0, accuracy=1.0)


@pytest.fixture
def tiny_net() -> TinyNet:
    return TinyNet()


@pytest.fixture
def classification_data() -> DataLoader:
    x = torch.randn(96, 8)
    y = (x[:, 0] > 0).long() + (x[:, 1] > 0).long()  # 3 separable-ish classes
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=False)


@pytest.fixture
def trainer(classification_data: DataLoader) -> ClassificationTrainer:
    return ClassificationTrainer(
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=classification_data,
        test_loader=classification_data,
    )
