"""Plain ``state_dict`` checkpoints (TorchScript is in maintenance mode upstream).

A checkpoint stores the pruned model's state (``*_orig`` weights plus ``*_mask``
buffers), the rewind snapshot, the number of completed rounds, the round history, the
search settings that must match on resume and the CPU and CUDA RNG states. Everything is a tensor
or a primitive, so it loads with ``torch.load(weights_only=True)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NotRequired, TypedDict

import torch
from torch import nn

from lottery.pruning import ParameterSelector, attach_masks, default_prunable_parameters

FORMAT_VERSION = 2
READABLE_VERSIONS = (1, 2)


class MetricsRecord(TypedDict):
    loss: float
    accuracy: float


class EpochRecord(TypedDict):
    epoch: int
    train: MetricsRecord
    test: MetricsRecord
    val: NotRequired[MetricsRecord | None]
    """Absent from histories written before validation metrics existed."""


class LayerRecord(TypedDict):
    name: str
    remaining: int
    total: int


class RoundRecord(TypedDict):
    """One :class:`~lottery.RoundResult`, as stored in a checkpoint's history."""

    round: int
    density: float
    epochs: list[EpochRecord]
    layers: list[LayerRecord]
    extra_metrics: dict[str, MetricsRecord]


class SearchConfig(TypedDict):
    """The search settings a checkpoint must agree with to be resumed."""

    rewind: str
    rewind_step: int
    strategy: str


@dataclass(frozen=True, slots=True)
class Checkpoint:
    rounds_completed: int
    rewind_state: dict[str, torch.Tensor] | None
    history: list[RoundRecord] = field(default_factory=list)
    config: SearchConfig | None = None
    """The search settings it was saved with; ``None`` for version 1 checkpoints."""
    rng_state: torch.Tensor | None = None
    """``torch.get_rng_state()`` at save time; ``None`` for version 1 checkpoints."""
    cuda_rng_state: list[torch.Tensor] | None = None
    """``torch.cuda.get_rng_state_all()`` at save time, one per GPU; ``None`` if saved
    without CUDA."""


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    rewind_state: dict[str, torch.Tensor] | None,
    rounds_completed: int,
    history: list[RoundRecord] | None = None,
    config: SearchConfig | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "model": model.state_dict(),
            "rewind_state": rewind_state,
            "rounds_completed": rounds_completed,
            "history": history or [],
            "config": config,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    parameters: ParameterSelector = default_prunable_parameters,
    map_location: torch.device | str | None = None,
) -> Checkpoint:
    """Restore weights and masks into ``model`` (built with the same architecture)."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    version = payload.get("format_version")
    if version not in READABLE_VERSIONS:
        raise ValueError(f"unsupported checkpoint format version: {version!r}")
    attach_masks(list(parameters(model)))
    model.load_state_dict(payload["model"])
    return Checkpoint(
        rounds_completed=int(payload["rounds_completed"]),
        rewind_state=payload["rewind_state"],
        history=list(payload.get("history", [])),
        config=payload.get("config"),
        rng_state=payload.get("rng_state"),
        cuda_rng_state=payload.get("cuda_rng_state"),
    )
