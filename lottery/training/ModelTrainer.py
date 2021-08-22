from abc import ABC, abstractmethod

from lottery.metrics.EpochMetrics import EpochMetrics
from lottery.metrics.ResultTypes import TrainTestResult


class ModelTrainer(ABC):
    @abstractmethod
    def train(self, _model, optimiser) -> EpochMetrics:
        pass

    @abstractmethod
    def test(self, _model) -> EpochMetrics:
        pass

    @abstractmethod
    def train_and_test(self, _model, training_iterations) -> TrainTestResult:
        pass
