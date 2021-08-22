from dataclasses import dataclass
from typing import List

import torch.nn as nn

from lottery.metrics.EpochMetrics import EpochMetrics


@dataclass
class EpochResult:
    train_result: EpochMetrics
    test_result: EpochMetrics


@dataclass
class TrainTestResult:
    results: List[EpochResult]
    model: nn.Module
