"""Iterative magnitude pruning (IMP) search for winning tickets."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lottery.callbacks import Callback
from lottery.checkpoint import load_checkpoint, save_checkpoint
from lottery.pruning import (
    GlobalMagnitudePruning,
    LayerSparsity,
    ParameterSelector,
    PrunableParameter,
    PruningStrategy,
    attach_masks,
    default_prunable_parameters,
    mask_keys,
    overall_density,
    sparsity_report,
)
from lottery.training import EpochResult, Metrics, Trainer

log = logging.getLogger(__name__)

type StateDict = dict[str, torch.Tensor]


class Rewind(StrEnum):
    """What happens to the surviving weights between pruning rounds."""

    WEIGHTS = "weights"
    """Reset to the snapshot taken at ``rewind_step`` (step 0 is the original init).
    This is the lottery ticket procedure."""
    RANDOM = "random"
    """Re-draw a fresh random initialisation. The random-reinit control experiment."""
    NONE = "none"
    """Keep the trained weights and retrain with a fresh optimiser and schedule
    (learning-rate rewinding, Renda et al. 2020)."""


@dataclass(frozen=True, slots=True)
class RoundResult:
    round: int
    density: float
    """Fraction of prunable weights still alive while this round trained."""
    epochs: list[EpochResult]
    layers: list[LayerSparsity]
    extra_metrics: dict[str, Metrics] = field(default_factory=dict)

    @property
    def final_test(self) -> Metrics | None:
        return self.epochs[-1].test if self.epochs else None


@dataclass(frozen=True, slots=True)
class SearchResult:
    rounds: list[RoundResult]

    def best(
        self,
        tolerance: float = 0.005,
        metric: Callable[[RoundResult], float | None] | None = None,
    ) -> RoundResult:
        """The sparsest round scoring within ``tolerance`` of the dense round (round 0).

        ``metric`` scores a round, higher is better; it defaults to the final epoch's test
        accuracy, and ``tolerance`` is in its units (0.005 is half a point of a 0-1
        accuracy). For a QAT search, score the real quantised model with
        ``metric=lambda r: r.extra_metrics["quantised"].accuracy``.

        Picking rounds by the same data you report on is optimistic. If that matters,
        give the trainer a validation loader as ``test_loader`` and evaluate the chosen
        ticket on held-out data afterwards.
        """
        if not self.rounds:
            raise ValueError("no rounds recorded")
        score = metric or _final_test_accuracy
        dense = next((r for r in self.rounds if r.round == 0), None)
        if dense is None:
            raise ValueError("the dense round (round 0) is not in this result")
        baseline = score(dense)
        if baseline is None or math.isnan(baseline):
            raise ValueError("the dense round has no score to compare the others against")
        eligible = [
            r for r in self.rounds if (s := score(r)) is not None and s >= baseline - tolerance
        ]
        return min(eligible, key=lambda r: r.density)


def _final_test_accuracy(result: RoundResult) -> float | None:
    return result.final_test.accuracy if result.final_test is not None else None


def rounds_for_density(target_density: float, prune_fraction: float) -> int:
    """Pruning rounds needed so that ``(1 - prune_fraction) ** rounds <= target_density``."""
    if not 0.0 < target_density <= 1.0:
        raise ValueError(f"target_density must be in (0, 1], got {target_density}")
    if not 0.0 < prune_fraction < 1.0:
        raise ValueError(f"prune_fraction must be in (0, 1), got {prune_fraction}")
    # The epsilon guards against float error turning an exact power into one extra round.
    return max(0, math.ceil(math.log(target_density) / math.log(1.0 - prune_fraction) - 1e-9))


def _reset_parameters(model: nn.Module) -> None:
    """Re-run each layer's own initialiser (the PyTorch default init for that layer type)."""
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()


class WinningTicket:
    """Search for a sparse, trainable sub-network by iterative magnitude pruning.

    Round 0 trains the dense network. Each later round prunes ``prune_fraction`` of the
    surviving weights by magnitude, rewinds the survivors (see :class:`Rewind`) and
    trains again.
    """

    def __init__(
        self,
        model: nn.Module,
        trainer: Trainer,
        *,
        strategy: PruningStrategy | None = None,
        parameters: ParameterSelector = default_prunable_parameters,
        rewind: Rewind | str = Rewind.WEIGHTS,
        rewind_step: int = 0,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int | None = None,
        callbacks: Sequence[Callback] = (),
    ) -> None:
        if rewind_step < 0:
            raise ValueError("rewind_step must be >= 0")
        if rewind_step > 0 and Rewind(rewind) is not Rewind.WEIGHTS:
            raise ValueError("rewind_step only applies to rewind='weights'")
        self.model = model
        self.trainer = trainer
        self.strategy: PruningStrategy = strategy or GlobalMagnitudePruning()
        self.rewind = Rewind(rewind)
        self.rewind_step = rewind_step
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.checkpoint_every = checkpoint_every
        self.callbacks = list(callbacks)
        if checkpoint_every is not None and checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")
        if checkpoint_every is not None and self.checkpoint_dir is None:
            raise ValueError("checkpoint_every requires checkpoint_dir")

        self.parameters: list[PrunableParameter] = list(parameters(model))
        if not self.parameters:
            raise ValueError("model has no prunable parameters")
        if self.rewind is Rewind.RANDOM:
            for module, name in self.parameters:
                if not callable(getattr(module, "reset_parameters", None)):
                    raise ValueError(
                        f"rewind='random' re-draws {type(module).__name__}.{name} with the "
                        "module's own reset_parameters(), which it does not have"
                    )
        attach_masks(self.parameters)
        self._rewind_state: StateDict | None = None
        if self.rewind is Rewind.WEIGHTS and rewind_step == 0:
            self._rewind_state = self._snapshot()
        self.rounds_completed = 0
        self.history: list[RoundResult] = []

    # ------------------------------------------------------------------ public API

    def search(self, rounds: int, epochs: int, prune_fraction: float = 0.2) -> SearchResult:
        """Run ``rounds`` pruning rounds, training ``epochs`` per round.

        A fresh search first trains the dense network (round 0), so it records
        ``rounds + 1`` results. Calling ``search`` again continues pruning from where
        the previous call stopped.
        """
        if rounds < 0 or epochs < 1:
            raise ValueError("rounds must be >= 0 and epochs >= 1")
        # A fresh search also trains the dense network as round 0.
        stop = self.rounds_completed + rounds + (1 if self.rounds_completed == 0 else 0)
        return self._run(
            epochs,
            prune_fraction,
            expected_rounds=stop - self.rounds_completed,
            keep_going=lambda: self.rounds_completed < stop,
        )

    def search_to_density(
        self, target_density: float, epochs: int, prune_fraction: float = 0.2
    ) -> SearchResult:
        """Prune one round at a time until at most ``target_density`` of the weights remain."""
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        # Only a hint for progress bars: exact for global pruning from a fresh start.
        pruned_rounds = max(self.rounds_completed - 1, 0)
        dense_round = 1 if self.rounds_completed == 0 else 0
        expected = max(rounds_for_density(target_density, prune_fraction) - pruned_rounds, 0)
        previous: float | None = None

        def keep_going() -> bool:
            nonlocal previous
            if self.rounds_completed == 0:
                return True
            density = self.density()
            if density <= target_density:
                return False
            if previous is not None and density >= previous:
                raise RuntimeError("pruning strategy removed no weights; cannot reach target")
            previous = density
            return True

        return self._run(
            epochs, prune_fraction, expected_rounds=expected + dense_round, keep_going=keep_going
        )

    def density(self) -> float:
        return overall_density(sparsity_report(self.model, self.parameters))

    def sparsity(self) -> list[LayerSparsity]:
        return sparsity_report(self.model, self.parameters)

    def masks(self) -> StateDict:
        """The current pruning masks, keyed by state-dict name."""
        keys = mask_keys(self.model)
        return {k: v.clone() for k, v in self.model.state_dict().items() if k in keys}

    def rewind_state(self) -> StateDict | None:
        """The weights survivors are reset to between rounds (``None`` until captured)."""
        return None if self._rewind_state is None else dict(self._rewind_state)

    def save(self, path: str | Path) -> Path:
        return save_checkpoint(
            path,
            model=self.model,
            rewind_state=self._rewind_state,
            rounds_completed=self.rounds_completed,
            history=[_round_to_dict(r) for r in self.history],
            config=self._config(),
        )

    def load(self, path: str | Path, map_location: torch.device | str | None = None) -> None:
        """Resume from a checkpoint written by :meth:`save` for the same architecture.

        Tensors are loaded onto the device the model currently lives on unless
        ``map_location`` says otherwise.
        """
        if map_location is None:
            map_location = next(self.model.parameters()).device
        checkpoint = load_checkpoint(
            path, self.model, parameters=lambda _: self.parameters, map_location=map_location
        )
        late_snapshot_pending = self.rewind_step > 0 and checkpoint.rounds_completed == 0
        if (
            self.rewind is Rewind.WEIGHTS
            and checkpoint.rewind_state is None
            and not late_snapshot_pending
        ):
            raise ValueError(
                "checkpoint has no rewind state, so it cannot resume a rewind='weights' search"
            )
        if len(checkpoint.history) != checkpoint.rounds_completed:
            raise ValueError(
                f"checkpoint has {len(checkpoint.history)} rounds of history but "
                f"rounds_completed={checkpoint.rounds_completed}, so it cannot resume a search"
            )
        if checkpoint.config is not None:
            self._check_config(checkpoint.config)
        self._rewind_state = checkpoint.rewind_state
        self.rounds_completed = checkpoint.rounds_completed
        self.history = [_round_from_dict(r) for r in checkpoint.history]
        if checkpoint.rng_state is not None:
            # So the rounds after a resume draw the same random numbers as an uninterrupted
            # run would (random re-init, shuffling); CUDA generators are not restored.
            torch.set_rng_state(checkpoint.rng_state.cpu())

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"WinningTicket(density={self.density():.4f}, rounds={self.rounds_completed})"

    # ------------------------------------------------------------------ hooks

    def _round_metrics(self) -> dict[str, Metrics]:
        """Extra per-round metrics; subclasses can add evaluations here."""
        return {}

    # ------------------------------------------------------------------ internals

    def _config(self) -> dict[str, Any]:
        return {
            "rewind": str(self.rewind),
            "rewind_step": self.rewind_step,
            "strategy": type(self.strategy).__qualname__,
        }

    def _check_config(self, saved: dict[str, Any]) -> None:
        mine = self._config()
        if (saved["rewind"], saved["rewind_step"]) != (mine["rewind"], mine["rewind_step"]):
            raise ValueError(
                f"checkpoint was saved with rewind='{saved['rewind']}', "
                f"rewind_step={saved['rewind_step']}, but this ticket uses "
                f"rewind='{mine['rewind']}', rewind_step={mine['rewind_step']}"
            )
        if saved["strategy"] != mine["strategy"]:
            warnings.warn(
                f"checkpoint was pruned with {saved['strategy']}, "
                f"this ticket continues with {mine['strategy']}",
                stacklevel=3,
            )

    def _run(
        self,
        epochs: int,
        prune_fraction: float,
        expected_rounds: int | None,
        keep_going: Callable[[], bool],
    ) -> SearchResult:
        # Checked here rather than by the strategy, which only sees it after round 0 trained.
        if not 0.0 < prune_fraction < 1.0:
            raise ValueError(f"prune_fraction must be in (0, 1), got {prune_fraction}")
        start = self.rounds_completed
        for callback in self.callbacks:
            callback.on_search_start(self, expected_rounds)
        try:
            while keep_going():
                result = self._train_round(self.rounds_completed, epochs, prune_fraction)
                for callback in self.callbacks:
                    callback.on_round_end(self, result)
        finally:
            for callback in self.callbacks:
                callback.on_search_end(self)
        last = self.rounds_completed - 1
        if self.checkpoint_dir is not None and last >= start and not self._checkpoint_due(last):
            self._checkpoint(last)
        return SearchResult(list(self.history))

    def _train_round(self, round_: int, epochs: int, prune_fraction: float) -> RoundResult:
        if round_ > 0:
            self.strategy.prune(self.parameters, prune_fraction)
            self._rewind()
        layers = sparsity_report(self.model, self.parameters)
        density = overall_density(layers)
        if self._needs_late_snapshot():
            epochs_result = self._fit_capturing_late_snapshot(epochs)
        else:
            epochs_result = self.trainer.fit(self.model, epochs)

        result = RoundResult(
            round=round_,
            density=density,
            epochs=epochs_result,
            layers=layers,
            extra_metrics=self._round_metrics(),
        )
        self.history.append(result)
        self.rounds_completed = round_ + 1
        log.info("%s", _describe(result))
        if self._checkpoint_due(round_):
            self._checkpoint(round_)
        return result

    def _checkpoint_due(self, round_: int) -> bool:
        return self.checkpoint_every is not None and round_ % self.checkpoint_every == 0

    def _checkpoint(self, round_: int) -> None:
        assert self.checkpoint_dir is not None
        path = self.save(self.checkpoint_dir / f"round_{round_:03d}.pt")
        log.debug("saved checkpoint %s", path)

    def _snapshot(self) -> StateDict:
        keys = mask_keys(self.model)
        return {k: v.detach().clone() for k, v in self.model.state_dict().items() if k not in keys}

    def _needs_late_snapshot(self) -> bool:
        return self.rewind is Rewind.WEIGHTS and self._rewind_state is None

    def _capture_rewind(self, step: int, model: nn.Module) -> None:
        if step == self.rewind_step and self._rewind_state is None:
            self._rewind_state = self._snapshot()

    def _fit_capturing_late_snapshot(self, epochs: int) -> list[EpochResult]:
        steps = _steps_per_epoch(self.trainer)
        if steps is not None and steps * epochs < self.rewind_step:
            raise ValueError(
                f"rewind_step={self.rewind_step} is never reached: the dense round trains "
                f"only {steps * epochs} steps ({epochs} epochs of {steps})"
            )
        initial = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        results = self.trainer.fit(self.model, epochs, on_step=self._capture_rewind)
        if self._rewind_state is None:
            # Put the untrained weights back so a retry starts from the real init.
            self.model.load_state_dict(initial)
            raise ValueError(
                f"rewind_step={self.rewind_step} was never reached during the dense round"
            )
        return results

    def _rewind(self) -> None:
        match self.rewind:
            case Rewind.WEIGHTS:
                if self._rewind_state is None:
                    raise RuntimeError("no rewind state captured")
                with torch.no_grad():
                    state = self.model.state_dict()
                    for key, value in self._rewind_state.items():
                        state[key].copy_(value)
            case Rewind.RANDOM:
                with torch.no_grad():
                    # On a pruned module `weight` is a derived tensor, so reset_parameters()
                    # re-draws that rather than the `weight_orig` parameter training updates.
                    # Copy each fresh draw across, whatever the layer's own init is.
                    _reset_parameters(self.model)
                    for module, name in self.parameters:
                        getattr(module, f"{name}_orig").copy_(getattr(module, name))
            case Rewind.NONE:
                pass


def _steps_per_epoch(trainer: Trainer) -> int | None:
    """Optimiser steps per epoch, if the trainer has a sized ``train_loader``."""
    try:
        return len(trainer.train_loader)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return None


def _describe(result: RoundResult) -> str:
    """One log line per round: ``round 3  density 51.20%  test acc 0.9712  quantised 0.9650``."""
    parts = [f"round {result.round}", f"density {result.density:.2%}"]
    if result.final_test is not None:
        parts.append(f"test acc {result.final_test.accuracy:.4f}")
    parts += [f"{name} {m.accuracy:.4f}" for name, m in result.extra_metrics.items()]
    return "  ".join(parts)


def _round_to_dict(result: RoundResult) -> dict[str, Any]:
    return asdict(result)


def _metrics(data: dict[str, float]) -> Metrics:
    return Metrics(loss=float(data["loss"]), accuracy=float(data["accuracy"]))


def _round_from_dict(data: dict[str, Any]) -> RoundResult:
    return RoundResult(
        round=int(data["round"]),
        density=float(data["density"]),
        epochs=[
            EpochResult(epoch=int(e["epoch"]), train=_metrics(e["train"]), test=_metrics(e["test"]))
            for e in data["epochs"]
        ],
        layers=[LayerSparsity(**layer) for layer in data["layers"]],
        extra_metrics={k: _metrics(v) for k, v in data["extra_metrics"].items()},
    )
