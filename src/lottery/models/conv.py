"""The Conv-2/4/6 CIFAR-10 networks from the LTH paper (scaled-down VGG variants)."""

import torch
from torch import nn


def _block(in_channels: int, out_channels: int) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    ]


class _ConvNet(nn.Module):
    def __init__(self, widths: list[int], num_classes: int, image_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for width in widths:
            layers += _block(in_channels, width)
            in_channels = width
        self.features = nn.Sequential(*layers)
        spatial = image_size // (2 ** len(widths))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * spatial * spatial, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))  # type: ignore[no-any-return]


class Conv2(_ConvNet):
    def __init__(self, num_classes: int = 10, image_size: int = 32) -> None:
        super().__init__([64], num_classes, image_size)


class Conv4(_ConvNet):
    def __init__(self, num_classes: int = 10, image_size: int = 32) -> None:
        super().__init__([64, 128], num_classes, image_size)


class Conv6(_ConvNet):
    def __init__(self, num_classes: int = 10, image_size: int = 32) -> None:
        super().__init__([64, 128, 256], num_classes, image_size)
