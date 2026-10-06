"""Training loops used inside each pruning round."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

type StepCallback = Callable[[int, nn.Module], None]
"""Called after every optimiser step with the global step count (1-based)."""
type EpochCallback = Callable[[EpochResult], None]
"""Called after every epoch, once its metrics are known."""
type OptimiserFactory = Callable[[Iterable[nn.Parameter]], Optimizer]
type SchedulerFactory = Callable[[Optimizer, int], LRScheduler]
"""Given the optimiser and the number of epochs in the round, build a per-epoch scheduler."""


@dataclass(frozen=True, slots=True)
class Metrics:
    loss: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    train: Metrics
    test: Metrics
    val: Metrics | None = None
    """Validation metrics, when the trainer has a validation loader."""


class Trainer(Protocol):
    """What :class:`~lottery.WinningTicket` needs from a trainer.

    ``fit`` may also accept an ``on_epoch: EpochCallback | None`` keyword. If it does,
    the search passes one so callbacks hear about every epoch; trainers without it still
    work, they just report once per round.
    """

    def fit(
        self, model: nn.Module, epochs: int, on_step: StepCallback | None = None
    ) -> list[EpochResult]: ...

    def evaluate(self, model: nn.Module, device: torch.device | None = None) -> Metrics: ...


class EpochReportingTrainer(Trainer, Protocol):
    """A :class:`Trainer` whose ``fit`` also reports each epoch as it finishes."""

    def fit(
        self,
        model: nn.Module,
        epochs: int,
        on_step: StepCallback | None = None,
        on_epoch: EpochCallback | None = None,
    ) -> list[EpochResult]: ...


def sgd(lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 5e-4) -> OptimiserFactory:
    def factory(params: Iterable[nn.Parameter]) -> Optimizer:
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    return factory


def adam(lr: float = 1e-3, weight_decay: float = 0.0) -> OptimiserFactory:
    def factory(params: Iterable[nn.Parameter]) -> Optimizer:
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    return factory


def cosine_annealing(optimiser: Optimizer, epochs: int) -> LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(epochs, 1))


@dataclass(slots=True)
class _Accumulator:
    loss_sum: float = 0.0
    correct: int = 0
    seen: int = 0

    def update(self, loss: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        batch = targets.shape[0]
        self.loss_sum += loss.item() * batch
        self.correct += int((outputs.argmax(dim=1) == targets).sum().item())
        self.seen += batch

    def result(self) -> Metrics:
        if self.seen == 0:
            return Metrics(loss=float("nan"), accuracy=float("nan"))
        return Metrics(loss=self.loss_sum / self.seen, accuracy=self.correct / self.seen)


@dataclass
class ClassificationTrainer:
    """Supervised classification trainer.

    A fresh optimiser (and scheduler, if configured) is built at the start of every
    ``fit`` call, so each pruning round starts from the same optimisation schedule.

    ``loss_fn`` must reduce to a scalar mean over the batch (the default for
    ``nn.CrossEntropyLoss``); the reported loss is the sample-weighted mean of it.
    """

    loss_fn: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    train_loader: DataLoader[Any]
    test_loader: DataLoader[Any]
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    optimiser: OptimiserFactory = field(default_factory=sgd)
    scheduler: SchedulerFactory | None = None
    autocast_dtype: torch.dtype | None = None
    """Mixed precision, for example ``torch.bfloat16``. ``None`` trains in full precision."""
    val_loader: DataLoader[Any] | None = None
    """Evaluated every epoch alongside ``test_loader``. When set, ``SearchResult.best``
    picks rounds by validation accuracy, keeping the test set out of the selection."""

    def fit(
        self,
        model: nn.Module,
        epochs: int,
        on_step: StepCallback | None = None,
        on_epoch: EpochCallback | None = None,
    ) -> list[EpochResult]:
        model.to(self.device)
        optimiser = self.optimiser(p for p in model.parameters() if p.requires_grad)
        scheduler = self.scheduler(optimiser, epochs) if self.scheduler else None
        results: list[EpochResult] = []
        step = 0
        for epoch in range(epochs):
            model.train()
            acc = _Accumulator()
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimiser.zero_grad(set_to_none=True)
                with self._autocast():
                    outputs = model(inputs)
                    loss = self.loss_fn(outputs, targets)
                loss.backward()
                optimiser.step()
                step += 1
                acc.update(loss.detach(), outputs.detach(), targets)
                if on_step is not None:
                    on_step(step, model)
            if scheduler is not None:
                scheduler.step()
            val = None if self.val_loader is None else self._evaluate(model, self.val_loader)
            result = EpochResult(
                epoch=epoch, train=acc.result(), test=self.evaluate(model), val=val
            )
            log.debug(
                "epoch %d train loss %.4f acc %.4f | %stest loss %.4f acc %.4f",
                epoch,
                result.train.loss,
                result.train.accuracy,
                "" if val is None else f"val loss {val.loss:.4f} acc {val.accuracy:.4f} | ",
                result.test.loss,
                result.test.accuracy,
            )
            results.append(result)
            if on_epoch is not None:
                on_epoch(result)
        return results

    def evaluate(self, model: nn.Module, device: torch.device | None = None) -> Metrics:
        return self._evaluate(model, self.test_loader, device)

    @torch.inference_mode()
    def _evaluate(
        self, model: nn.Module, loader: DataLoader[Any], device: torch.device | None = None
    ) -> Metrics:
        device = device or self.device
        model.to(device)
        model.eval()
        acc = _Accumulator()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with self._autocast(device):
                outputs = model(inputs)
                loss = self.loss_fn(outputs, targets)
            acc.update(loss, outputs, targets)
        return acc.result()

    def _autocast(self, device: torch.device | None = None) -> torch.autocast | nullcontext[None]:
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast((device or self.device).type, dtype=self.autocast_dtype)
