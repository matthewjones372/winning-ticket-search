import torch
import torch.nn as nn


class DenseNetworkQAT(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNetworkQAT, self).__init__()
        self.classifier = nn.Sequential(
            torch.quantization.QuantStub(),
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
            torch.quantization.DeQuantStub(),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
