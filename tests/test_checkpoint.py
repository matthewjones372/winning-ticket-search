import csv

import pytest
import torch

from lottery import CsvLogger, WinningTicket, load_checkpoint, save_checkpoint

from .conftest import ShiftTrainer, TinyNet


def test_round_trip_restores_weights_masks_and_rewind_state(tmp_path):
    ticket = WinningTicket(TinyNet(), ShiftTrainer())
    ticket.search(rounds=2, epochs=1)
    path = ticket.save(tmp_path / "nested" / "ticket.pt")

    fresh = TinyNet()
    checkpoint = load_checkpoint(path, fresh)

    assert checkpoint.rounds_completed == 3
    assert checkpoint.rewind_state is not None
    for key, value in ticket.model.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value), key


def test_checkpoints_load_with_weights_only(tmp_path):
    path = save_checkpoint(
        tmp_path / "c.pt", model=TinyNet(), rewind_state=None, rounds_completed=0
    )
    payload = torch.load(path, weights_only=True)
    assert payload["format_version"] == 1


def test_unknown_format_version_rejected(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 99}, path)
    with pytest.raises(ValueError, match="format version"):
        load_checkpoint(path, TinyNet())


def test_ticket_resumes_from_checkpoint(tmp_path):
    original = WinningTicket(TinyNet(), ShiftTrainer())
    original.search(rounds=2, epochs=1)
    path = original.save(tmp_path / "t.pt")

    resumed = WinningTicket(TinyNet(), ShiftTrainer())
    resumed.load(path)
    assert resumed.rounds_completed == 3
    assert resumed.density() == pytest.approx(original.density())

    result = resumed.search(rounds=1, epochs=1)
    assert [r.round for r in result.rounds] == [0, 1, 2, 3], "history is restored"
    assert result.rounds[1] == original.history[1]
    assert resumed.density() == pytest.approx(0.8**3, abs=0.01)
    assert result.best().round == 3  # baseline is still the dense round


def test_resume_keeps_earlier_csv_rows(tmp_path):
    original = WinningTicket(TinyNet(), ShiftTrainer(), callbacks=[CsvLogger(tmp_path)])
    original.search(rounds=2, epochs=1)
    path = original.save(tmp_path / "t.pt")
    original.search(rounds=1, epochs=1)  # round 3, not in the checkpoint

    resumed = WinningTicket(TinyNet(), ShiftTrainer(), callbacks=[CsvLogger(tmp_path)])
    resumed.load(path)
    resumed.search(rounds=2, epochs=1)

    with (tmp_path / "metrics.csv").open() as fh:
        rounds = [int(row["round"]) for row in csv.DictReader(fh)]
    assert rounds == [0, 1, 2, 3, 4]


def test_loading_without_rewind_state_into_weights_ticket_is_rejected(tmp_path):
    source = WinningTicket(TinyNet(), ShiftTrainer(), rewind="random")
    source.search(rounds=1, epochs=1)
    path = source.save(tmp_path / "t.pt")

    with pytest.raises(ValueError, match="no rewind state"):
        WinningTicket(TinyNet(), ShiftTrainer()).load(path)

    resumed = WinningTicket(TinyNet(), ShiftTrainer(), rewind="random")
    resumed.load(path)
    resumed.search(rounds=1, epochs=1)
    assert resumed.rounds_completed == 3


def test_late_rewind_checkpoint_before_capture_can_be_loaded(tmp_path):
    source = WinningTicket(TinyNet(), ShiftTrainer(), rewind_step=2)
    path = source.save(tmp_path / "t.pt")
    resumed = WinningTicket(TinyNet(), ShiftTrainer(), rewind_step=2)
    resumed.load(path, map_location="cpu")
    resumed.search(rounds=1, epochs=1)
    assert resumed.rewind_state() is not None
