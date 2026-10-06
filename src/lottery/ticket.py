"""Iterative magnitude pruning (IMP) search for winning tickets."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from lottery.checkpoint import load_checkpoint, save_checkpoint
from lottery.pruning import (
    GlobalMagnitudePruning,
    LayerSparsity,
    ParameterSelector,
    PrunableParameter,
    PruningStrategy,
    attach_masks,
    default_prunable_parameters,
    overall_density,
    sparsity_report,
)
from lottery.reporting import CsvReporter
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

    def best(self) -> RoundResult:
        """The sparsest round whose final test accuracy is within 0.5 points of the dense run."""
        if not self.rounds:
            raise ValueError("no rounds recorded")
        dense = self.rounds[0].final_test
        if dense is None:
            return self.rounds[0]
        eligible = [
            r
            for r in self.rounds
            if r.final_test and r.final_test.accuracy >= dense.accuracy - 0.005
        ]
        return min(eligible, key=lambda r: r.density)


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
        output_dir: str | Path | None = None,
        checkpoint_every: int | None = None,
        progress: bool = True,
    ) -> None:
        if rewind_step < 0:
            raise ValueError("rewind_step must be >= 0")
        self.model = model
        self.trainer = trainer
        self.strategy: PruningStrategy = strategy or GlobalMagnitudePruning()
        self.rewind = Rewind(rewind)
        self.rewind_step = rewind_step
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.checkpoint_every = checkpoint_every
        self.progress = progress
        if checkpoint_every is not None and self.output_dir is None:
            raise ValueError("checkpoint_every requires output_dir")

        self.parameters: list[PrunableParameter] = list(parameters(model))
        if not self.parameters:
            raise ValueError("model has no prunable parameters")
        attach_masks(self.parameters)
        self._rewind_state: StateDict | None = None
        if self.rewind is Rewind.WEIGHTS and rewind_step == 0:
            self._rewind_state = self._snapshot()
        self.rounds_completed = 0
        self.history: list[RoundResult] = []
        self._reporter: CsvReporter | None = None

    # ------------------------------------------------------------------ public API

    def search(self, rounds: int, epochs: int, prune_fraction: float = 0.2) -> SearchResult:
        """Run ``rounds`` pruning rounds, training ``epochs`` per round.

        A fresh search first trains the dense network (round 0), so it records
        ``rounds + 1`` results. Calling ``search`` again continues pruning from where
        the previous call stopped.
        """
        if rounds < 0 or epochs < 1:
            raise ValueError("rounds must be >= 0 and epochs >= 1")
        if self._reporter is None and self.output_dir is not None:
            self._reporter = CsvReporter(self.output_dir)
        reporter = self._reporter
        start = self.rounds_completed
        # A fresh search also trains the dense network as round 0.
        stop = start + rounds + (1 if start == 0 else 0)
        bar = tqdm(range(start, stop), desc="pruning round", disable=not self.progress)
        for round_ in bar:
            if round_ > 0:
                self.strategy.prune(self.parameters, prune_fraction)
                self._rewind()
            layers = sparsity_report(self.model, self.parameters)
            density = overall_density(layers)
            bar.set_postfix(density=f"{density:.3f}")

            on_step = self._capture_rewind if self._needs_late_snapshot() else None
            epochs_result = self.trainer.fit(self.model, epochs, on_step=on_step)
            if self._needs_late_snapshot():
                raise ValueError(
                    f"rewind_step={self.rewind_step} was never reached during the dense round"
                )

            result = RoundResult(
                round=round_,
                density=density,
                epochs=epochs_result,
                layers=layers,
                extra_metrics=self._round_metrics(),
            )
            self.history.append(result)
            self.rounds_completed = round_ + 1
            if reporter is not None:
                reporter.write_round(result)
            if self.checkpoint_every and round_ % self.checkpoint_every == 0:
                self.save(self._checkpoint_path(round_))
            if result.final_test is not None:
                log.info(
                    "round %d density %.4f test acc %.4f",
                    round_,
                    density,
                    result.final_test.accuracy,
                )
        return SearchResult(list(self.history))

    def search_to_density(
        self, target_density: float, epochs: int, prune_fraction: float = 0.2
    ) -> SearchResult:
        """Prune until at most ``target_density`` of the prunable weights remain."""
        needed = rounds_for_density(target_density, prune_fraction)
        done = max(self.rounds_completed - 1, 0)
        return self.search(max(needed - done, 0), epochs, prune_fraction)

    def density(self) -> float:
        return overall_density(sparsity_report(self.model, self.parameters))

    def sparsity(self) -> list[LayerSparsity]:
        return sparsity_report(self.model, self.parameters)

    def masks(self) -> StateDict:
        """The current pruning masks, keyed by state-dict name."""
        return {k: v.clone() for k, v in self.model.state_dict().items() if k.endswith("_mask")}

    def rewind_state(self) -> StateDict | None:
        """The weights survivors are reset to between rounds (``None`` until captured)."""
        return None if self._rewind_state is None else dict(self._rewind_state)

    def save(self, path: str | Path) -> Path:
        return save_checkpoint(
            path,
            model=self.model,
            rewind_state=self._rewind_state,
            rounds_completed=self.rounds_completed,
        )

    def load(self, path: str | Path) -> None:
        """Resume from a checkpoint written by :meth:`save` for the same architecture."""
        checkpoint = load_checkpoint(path, self.model, parameters=lambda _: self.parameters)
        self._rewind_state = checkpoint.rewind_state
        self.rounds_completed = checkpoint.rounds_completed

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"WinningTicket(density={self.density():.4f}, rounds={self.rounds_completed})"

    # ------------------------------------------------------------------ hooks

    def _round_metrics(self) -> dict[str, Metrics]:
        """Extra per-round metrics; subclasses can add evaluations here."""
        return {}

    # ------------------------------------------------------------------ internals

    def _snapshot(self) -> StateDict:
        return {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
            if not k.endswith("_mask")
        }

    def _needs_late_snapshot(self) -> bool:
        return self.rewind is Rewind.WEIGHTS and self._rewind_state is None

    def _capture_rewind(self, step: int, model: nn.Module) -> None:
        if step == self.rewind_step and self._rewind_state is None:
            self._rewind_state = self._snapshot()

    def _rewind(self) -> None:
        match self.rewind:
            case Rewind.WEIGHTS:
                assert self._rewind_state is not None
                with torch.no_grad():
                    state = self.model.state_dict()
                    for key, value in self._rewind_state.items():
                        state[key].copy_(value)
            case Rewind.RANDOM:
                with torch.no_grad():
                    # reset_parameters() re-draws biases and norm layers, but on a pruned
                    # module `weight` is a derived tensor, so re-draw `weight_orig` directly
                    # with the same default init PyTorch uses for linear and conv layers.
                    _reset_parameters(self.model)
                    for module, name in self.parameters:
                        nn.init.kaiming_uniform_(getattr(module, f"{name}_orig"), a=math.sqrt(5))
            case Rewind.NONE:
                pass

    def _checkpoint_path(self, round_: int) -> Path:
        assert self.output_dir is not None
        return self.output_dir / "checkpoints" / f"round_{round_:03d}.pt"
