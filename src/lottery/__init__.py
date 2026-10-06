"""Lottery ticket search by iterative magnitude pruning."""

from importlib.metadata import PackageNotFoundError, version

from lottery.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from lottery.pruning import (
    GlobalMagnitudePruning,
    LayerSparsity,
    LayerwiseMagnitudePruning,
    PruningStrategy,
    default_prunable_parameters,
)
from lottery.ticket import Rewind, RoundResult, SearchResult, WinningTicket, rounds_for_density
from lottery.training import (
    ClassificationTrainer,
    EpochResult,
    Metrics,
    Trainer,
    adam,
    cosine_annealing,
    sgd,
)

try:
    __version__ = version("lottery")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "Checkpoint",
    "ClassificationTrainer",
    "EpochResult",
    "GlobalMagnitudePruning",
    "LayerSparsity",
    "LayerwiseMagnitudePruning",
    "Metrics",
    "PruningStrategy",
    "Rewind",
    "RoundResult",
    "SearchResult",
    "Trainer",
    "WinningTicket",
    "__version__",
    "adam",
    "cosine_annealing",
    "default_prunable_parameters",
    "load_checkpoint",
    "rounds_for_density",
    "save_checkpoint",
    "sgd",
]
