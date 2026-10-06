import pytest
import torch

from lottery import WinningTicket, load_checkpoint, save_checkpoint

from .conftest import ShiftTrainer, TinyNet


def test_round_trip_restores_weights_masks_and_rewind_state(tmp_path):
    ticket = WinningTicket(TinyNet(), ShiftTrainer(), progress=False)
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
    original = WinningTicket(TinyNet(), ShiftTrainer(), progress=False)
    original.search(rounds=2, epochs=1)
    path = original.save(tmp_path / "t.pt")

    resumed = WinningTicket(TinyNet(), ShiftTrainer(), progress=False)
    resumed.load(path)
    assert resumed.rounds_completed == 3
    assert resumed.density() == pytest.approx(original.density())

    result = resumed.search(rounds=1, epochs=1)
    assert [r.round for r in result.rounds] == [3]
    assert resumed.density() == pytest.approx(0.8**3, abs=0.01)
