import torch
import torchmetrics
from dataclasses import dataclass


@dataclass
class EpochMetrics(object):
    accuracy: float = 0.0
    loss: float = 0.0

    def __init__(self, device):
        self.accuracy_metric = torchmetrics.Accuracy().to(device)
        self.loss_metric = torchmetrics.MeanAbsoluteError().to(device)

    def update_batch(self, y_pred, y_true) -> None:
        _, predicted_labels = torch.max(y_pred.data, dim=1)

        self.accuracy_metric(predicted_labels, y_true)
        self.loss_metric(predicted_labels, y_true)

    def compute(self):
        self.accuracy = self.accuracy_metric.compute().item()
        self.loss = self.loss_metric.compute().item()
        return self
