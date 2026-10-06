"""CSV output for search results."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lottery.ticket import RoundResult

METRICS_FIELDS = [
    "round",
    "density",
    "epoch",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
]
LAYER_FIELDS = ["round", "layer", "remaining", "total", "density"]
EXTRA_FIELDS = ["round", "name", "loss", "accuracy"]


class CsvReporter:
    """Appends one round at a time to ``metrics.csv``, ``layers.csv`` and ``extra.csv``.

    Files are truncated when the reporter is created, so a new search does not mix
    rows with a previous run in the same directory.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.directory / "metrics.csv"
        self.layers_path = self.directory / "layers.csv"
        self.extra_path = self.directory / "extra.csv"
        for path, fields in (
            (self.metrics_path, METRICS_FIELDS),
            (self.layers_path, LAYER_FIELDS),
            (self.extra_path, EXTRA_FIELDS),
        ):
            with path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=fields).writeheader()

    def write_round(self, result: RoundResult) -> None:
        self._append(
            self.metrics_path,
            METRICS_FIELDS,
            (
                {
                    "round": result.round,
                    "density": result.density,
                    "epoch": e.epoch,
                    "train_loss": e.train.loss,
                    "train_accuracy": e.train.accuracy,
                    "test_loss": e.test.loss,
                    "test_accuracy": e.test.accuracy,
                }
                for e in result.epochs
            ),
        )
        self._append(
            self.layers_path,
            LAYER_FIELDS,
            (
                {
                    "round": result.round,
                    "layer": layer.name,
                    "remaining": layer.remaining,
                    "total": layer.total,
                    "density": layer.density,
                }
                for layer in result.layers
            ),
        )
        self._append(
            self.extra_path,
            EXTRA_FIELDS,
            (
                {"round": result.round, "name": name, "loss": m.loss, "accuracy": m.accuracy}
                for name, m in result.extra_metrics.items()
            ),
        )

    @staticmethod
    def _append(path: Path, fields: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
        with path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerows(rows)
