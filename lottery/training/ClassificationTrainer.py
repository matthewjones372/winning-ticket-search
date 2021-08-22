from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
import tqdm

from lottery.metrics.EpochMetrics import EpochMetrics
from lottery.metrics.ResultTypes import TrainTestResult, EpochResult
from lottery.training.ModelTrainer import ModelTrainer
from lottery.training.OptimiserType import OptimiserType


class ClassificationTrainer(ModelTrainer):
    def __init__(
            self,
            loss_func,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            device: torch.device,
            enable_logging: bool = True,
            with_scheduler: bool = False,
            optimiser_type: OptimiserType = OptimiserType.SGD,
            is_quantised_model: bool = False,
    ):
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.logging = enable_logging
        self.with_scheduler = with_scheduler
        self.optimiser_type = optimiser_type
        self.is_quantised_model = is_quantised_model

    def train_and_test(
            self, _model: nn.Module, training_iterations: int
    ) -> TrainTestResult:
        optimiser = self.__load_optimiser(_model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=training_iterations
        )
        results: List[EpochResult] = []

        training_message = "################ Train/Test Iteration ########"
        training_loop = tqdm.tqdm(range(training_iterations), desc=training_message, position=2, leave=True)
        for _ in training_loop:
            train_result = self.train(_model, optimiser)
            test_result = self.test(_model)

            if self.with_scheduler:
                scheduler.step()

            if self.logging:
                print()
                print(
                    f"-> Training Accuracy: {train_result.accuracy:.3f}\n"
                    f"-> Training Loss: {train_result.loss:.3f}\n"
                    f"-> Test Accuracy: {test_result.accuracy:.3f}\n"
                    f"-> Test Loss: {test_result.loss:.3f}\n"
                )

                results.append(EpochResult(train_result, test_result))

        return TrainTestResult(results, _model)

    def train(self, _model: nn.Module, optimiser: torch.optim) -> EpochMetrics:
        _model.train()

        metrics = EpochMetrics(device=self.device)
        for data, target in self.train_loader:
            images, labels = data.to(self.device), target.to(self.device)
            optimiser.zero_grad()
            outputs = _model(images)
            loss = self.loss_func(outputs, labels)
            loss.backward()
            self.__zero_out_gradients(_model)
            optimiser.step()

            metrics.update_batch(outputs, labels)

        return metrics.compute()

    def test(self, _model):
        _model.eval()

        if self.is_quantised_model:
            model_int8 = deepcopy(_model)
            model_int8 = torch.quantization.convert(model_int8.to(torch.device("cpu")))
            with torch.no_grad():
                metrics = EpochMetrics(device=torch.device("cpu"))
                for (data, target) in self.test_loader:
                    images, labels = data.to(torch.device("cpu")), target.to(
                        torch.device("cpu")
                    )
                    outputs = model_int8(images)
                    metrics.update_batch(outputs, labels)
                return metrics.compute()

        else:
            with torch.no_grad():
                metrics = EpochMetrics(device=self.device)
                for (data, target) in self.test_loader:
                    images, labels = data.to(self.device), target.to(self.device)
                    outputs = _model(images)
                    metrics.update_batch(outputs, labels)

            return metrics.compute()

    def __load_optimiser(self, _model: nn.Module) -> torch.optim:
        if self.optimiser_type == OptimiserType.ADAM:
            return torch.optim.Adam(_model.parameters(), weight_decay=1e-4)

        elif self.optimiser_type == OptimiserType.SGD:
            return torch.optim.SGD(
                _model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
            )

        else:
            return torch.optim.Adam(_model.parameters())

    def __zero_out_gradients(self, _model: nn.Module):
        epsilon = 1e-6
        zero_scaler = torch.tensor(0.0).to(self.device)

        for name, p in _model.named_parameters():
            if "weight" in name:
                p.grad.data = torch.where(p.data < epsilon, zero_scaler, p.grad.data)
