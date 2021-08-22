import torch
import torch.nn as nn


class DenseNetwork(nn.Module):

    def __init__(self, num_classes=10):
        super(DenseNetwork, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
