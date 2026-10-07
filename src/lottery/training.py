"""Training loops used inside each pruning round."""

from __future__ import annotations

import itertools
import logging
import warnings
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
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
"""Builds the optimiser for one round's parameters. Any torch optimiser works through
``functools.partial``, e.g. ``partial(torch.optim.AdamW, lr=3e-4)``; :func:`sgd` and
:func:`adam` are shorthands with the defaults Frankle & Carbin used."""
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


class StatefulTrainer(Trainer, Protocol):
    """A trainer with random state of its own, such as a ``DataLoader`` seeded through
    ``generator=``, which checkpoints save and restore so a resumed search shuffles the
    same way as an uninterrupted one."""

    def state_dict(self) -> dict[str, torch.Tensor]: ...

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None: ...


class ResumableTrainer(EpochReportingTrainer, Protocol):
    """A trainer that can also start part-way through its schedule.

    Late rewinding (Frankle et al. 2020) resets the weights to those from step ``k`` and
    trains the remaining steps with the learning-rate schedule where it was at step
    ``k``. ``start_step=k`` asks ``fit`` for exactly that.
    """

    def fit(
        self,
        model: nn.Module,
        epochs: int,
        on_step: StepCallback | None = None,
        on_epoch: EpochCallback | None = None,
        start_step: int = 0,
    ) -> list[EpochResult]: ...


def sgd(lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 5e-4) -> OptimiserFactory:
    """``torch.optim.SGD`` with these settings. Shorthand for ``functools.partial``."""
    return partial(torch.optim.SGD, lr=lr, momentum=momentum, weight_decay=weight_decay)


def adam(lr: float = 1e-3, weight_decay: float = 0.0) -> OptimiserFactory:
    """``torch.optim.Adam`` with these settings. Shorthand for ``functools.partial``."""
    return partial(torch.optim.Adam, lr=lr, weight_decay=weight_decay)


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
    device: torch.device | str = field(default_factory=lambda: torch.device("cpu"))
    optimiser: OptimiserFactory = field(default_factory=sgd)
    scheduler: SchedulerFactory | None = None
    autocast_dtype: torch.dtype | None = None
    """Mixed precision, for example ``torch.bfloat16``. ``None`` trains in full precision."""
    val_loader: DataLoader[Any] | None = None
    """Evaluated every epoch alongside ``test_loader``. When set, ``SearchResult.best``
    picks rounds by validation accuracy, keeping the test set out of the selection."""

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """The states of the loaders' own random generators (see :class:`StatefulTrainer`)."""
        return {name: g.get_state() for name, g in self._generators().items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        generators = self._generators()
        for name, value in state.items():
            if name in generators:
                generators[name].set_state(value.cpu())

    def _generators(self) -> dict[str, torch.Generator]:
        """Each loader's generator, and its sampler's when that is a different one."""
        found: dict[str, torch.Generator] = {}
        loaders = {"train": self.train_loader, "test": self.test_loader, "val": self.val_loader}
        for name, loader in loaders.items():
            if loader is None:
                continue
            seen: set[int] = set()
            for part, owner in (("generator", loader), ("sampler.generator", loader.sampler)):
                generator = getattr(owner, "generator", None)
                if isinstance(generator, torch.Generator) and id(generator) not in seen:
                    seen.add(id(generator))
                    found[f"{name}.{part}"] = generator
        return found

    def fit(
        self,
        model: nn.Module,
        epochs: int,
        on_step: StepCallback | None = None,
        on_epoch: EpochCallback | None = None,
        start_step: int = 0,
    ) -> list[EpochResult]:
        """Train for ``epochs`` epochs, or for what remains of them after ``start_step``.

        With ``start_step=k`` the first ``k`` optimiser steps are skipped and the
        scheduler is advanced past the epochs they cover, so training picks the schedule
        up at step ``k``. Step numbers passed to ``on_step`` carry on from ``k``.
        """
        steps_per_epoch = len(self.train_loader)
        if not 0 <= start_step < epochs * steps_per_epoch:
            raise ValueError(
                f"start_step={start_step} is outside the {epochs * steps_per_epoch} steps "
                f"of {epochs} epochs"
            )
        start_epoch, skip = divmod(start_step, steps_per_epoch)
        model.to(self.device)
        optimiser = self.optimiser(p for p in model.parameters() if p.requires_grad)
        scheduler = self.scheduler(optimiser, epochs) if self.scheduler else None
        if scheduler is not None and start_epoch:
            with warnings.catch_warnings():
                # Stepping before any optimiser step is the point here.
                warnings.filterwarnings(
                    "ignore", "Detected call of `lr_scheduler.step", UserWarning
                )
                for _ in range(start_epoch):
                    scheduler.step()
        results: list[EpochResult] = []
        step = start_step
        for epoch in range(start_epoch, epochs):
            model.train()
            acc = _Accumulator()
            batches = iter(self.train_loader)
            if epoch == start_epoch:
                batches = itertools.islice(batches, skip, None)
            for inputs, targets in batches:
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
        device = torch.device(device or self.device)
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
        return torch.autocast(torch.device(device or self.device).type, dtype=self.autocast_dtype)
