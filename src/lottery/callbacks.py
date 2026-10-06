"""Hooks into a running search, and the ones that ship with the package.

The search itself only talks to :mod:`logging` (one ``INFO`` line per round from
``lottery.ticket``, one ``DEBUG`` line per epoch from ``lottery.training``) and is silent
until the application configures logging. Anything else -- progress bars, CSV files,
experiment trackers -- is a :class:`Callback` passed to ``WinningTicket(callbacks=...)``.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

if TYPE_CHECKING:
    from lottery.ticket import RoundResult, WinningTicket


class Callback:
    """Base class with no-op hooks. Override the ones you need.

    ``on_search_end`` is called even when a round raises, so it is the place to release
    anything ``on_search_start`` acquired.
    """

    def on_search_start(self, ticket: WinningTicket, rounds: int | None) -> None:
        """``rounds`` is how many rounds this call expects to run, or ``None`` if unknown."""

    def on_round_end(self, ticket: WinningTicket, result: RoundResult) -> None:
        pass

    def on_search_end(self, ticket: WinningTicket) -> None:
        pass


class ProgressBar(Callback):
    """A tqdm bar over pruning rounds.

    While it is open, records sent to the root logger's handlers are written through
    tqdm so they appear above the bar instead of breaking it.
    """

    def __init__(self, desc: str = "pruning round") -> None:
        self.desc = desc
        self._bar: tqdm[Any] | None = None
        self._stack = ExitStack()

    def on_search_start(self, ticket: WinningTicket, rounds: int | None) -> None:
        self._stack.enter_context(logging_redirect_tqdm())
        self._bar = self._stack.enter_context(tqdm(total=rounds, desc=self.desc))

    def on_round_end(self, ticket: WinningTicket, result: RoundResult) -> None:
        assert self._bar is not None
        postfix = {"density": f"{result.density:.3f}"}
        if result.final_test is not None:
            postfix["acc"] = f"{result.final_test.accuracy:.4f}"
        self._bar.set_postfix(postfix)
        self._bar.update()

    def on_search_end(self, ticket: WinningTicket) -> None:
        self._stack.close()
        self._bar = None


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


class CsvLogger(Callback):
    """Appends each round to ``metrics.csv``, ``layers.csv`` and ``extra.csv`` in ``directory``.

    The first time it sees a search, rows for rounds the ticket has not completed yet are
    dropped: a fresh search starts with empty files, and one resumed from a checkpoint
    keeps the history before it.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.metrics_path = self.directory / "metrics.csv"
        self.layers_path = self.directory / "layers.csv"
        self.extra_path = self.directory / "extra.csv"
        self._started = False

    def on_search_start(self, ticket: WinningTicket, rounds: int | None) -> None:
        if self._started:
            return
        self._started = True
        self.directory.mkdir(parents=True, exist_ok=True)
        keep_below = ticket.rounds_completed
        for path, fields in (
            (self.metrics_path, METRICS_FIELDS),
            (self.layers_path, LAYER_FIELDS),
            (self.extra_path, EXTRA_FIELDS),
        ):
            kept: list[dict[str, str]] = []
            if keep_below > 0 and path.exists():
                with path.open(newline="", encoding="utf-8") as fh:
                    kept = [r for r in csv.DictReader(fh) if int(r["round"]) < keep_below]
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(kept)

    def on_round_end(self, ticket: WinningTicket, result: RoundResult) -> None:
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
