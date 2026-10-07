"""Lottery ticket search by iterative magnitude pruning."""

import logging
from importlib.metadata import PackageNotFoundError, version

from lottery.callbacks import Callback, CsvLogger, ProgressBar
from lottery.checkpoint import Checkpoint, RoundRecord, load_checkpoint, save_checkpoint
from lottery.pruning import (
    GlobalMagnitudePruning,
    LayerSparsity,
    LayerwiseMagnitudePruning,
    ParameterSelector,
    PrunableParameter,
    PruningStrategy,
    default_prunable_parameters,
)
from lottery.ticket import (
    Rewind,
    RoundResult,
    SearchResult,
    TicketOptions,
    WinningTicket,
    rounds_for_density,
    train_with_masks,
)
from lottery.training import (
    ClassificationTrainer,
    EpochCallback,
    EpochReportingTrainer,
    EpochResult,
    Metrics,
    OptimiserFactory,
    ResumableTrainer,
    SchedulerFactory,
    StatefulTrainer,
    StepCallback,
    Trainer,
    adam,
    cosine_annealing,
    sgd,
)

try:
    __version__ = version("lottery")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

# Silent unless the application configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Callback",
    "Checkpoint",
    "ClassificationTrainer",
    "CsvLogger",
    "EpochCallback",
    "EpochReportingTrainer",
    "EpochResult",
    "GlobalMagnitudePruning",
    "LayerSparsity",
    "LayerwiseMagnitudePruning",
    "Metrics",
    "OptimiserFactory",
    "ParameterSelector",
    "ProgressBar",
    "PrunableParameter",
    "PruningStrategy",
    "ResumableTrainer",
    "Rewind",
    "RoundRecord",
    "RoundResult",
    "SchedulerFactory",
    "SearchResult",
    "StatefulTrainer",
    "StepCallback",
    "TicketOptions",
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
    "train_with_masks",
]
