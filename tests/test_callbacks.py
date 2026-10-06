import logging

import pytest

from lottery import Callback, ProgressBar, WinningTicket
from lottery.pruning import PruningStrategy

from .conftest import ShiftTrainer, TinyNet


class Recorder(Callback):
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def on_search_start(self, ticket, rounds):
        self.events.append(("start", rounds))

    def on_round_end(self, ticket, result):
        self.events.append(("round", result.round))

    def on_search_end(self, ticket):
        self.events.append(("end", ticket.rounds_completed))


def test_hooks_fire_once_per_search_and_round():
    recorder = Recorder()
    ticket = WinningTicket(TinyNet(), ShiftTrainer(), callbacks=[recorder])
    ticket.search(rounds=2, epochs=1)
    ticket.search(rounds=1, epochs=1)
    assert recorder.events == [
        ("start", 3),
        ("round", 0),
        ("round", 1),
        ("round", 2),
        ("end", 3),
        ("start", 1),
        ("round", 3),
        ("end", 4),
    ]


def test_search_to_density_is_one_search_with_a_round_estimate():
    recorder = Recorder()
    ticket = WinningTicket(TinyNet(), ShiftTrainer(), callbacks=[recorder])
    ticket.search_to_density(0.5, epochs=1)  # 0.8 ** 4 = 0.41 is the first <= 0.5
    starts = [e for e in recorder.events if e[0] == "start"]
    assert starts == [("start", 5)]
    assert [e[1] for e in recorder.events if e[0] == "round"] == [0, 1, 2, 3, 4]


def test_search_end_fires_when_a_round_raises():
    class Broken(PruningStrategy):
        def prune(self, parameters, fraction):
            raise RuntimeError("boom")

    recorder = Recorder()
    ticket = WinningTicket(TinyNet(), ShiftTrainer(), strategy=Broken(), callbacks=[recorder])
    with pytest.raises(RuntimeError, match="boom"):
        ticket.search(rounds=1, epochs=1)
    assert recorder.events == [("start", 2), ("round", 0), ("end", 1)]


def test_one_info_line_per_round(caplog):
    ticket = WinningTicket(TinyNet(), ShiftTrainer())
    with caplog.at_level(logging.INFO, logger="lottery"):
        ticket.search(rounds=1, epochs=1)
    lines = [r.getMessage() for r in caplog.records if r.name == "lottery.ticket"]
    assert lines == [
        "round 0  density 100.00%  test acc 0.5000",
        "round 1  density 80.11%  test acc 0.5000",
    ]


def test_progress_bar_closes_and_restores_logging(capsys):
    root_handlers = list(logging.getLogger().handlers)
    bar = ProgressBar()
    ticket = WinningTicket(TinyNet(), ShiftTrainer(), callbacks=[bar])
    ticket.search(rounds=2, epochs=1)
    assert bar._bar is None
    assert logging.getLogger().handlers == root_handlers
    assert "3/3" in capsys.readouterr().err
