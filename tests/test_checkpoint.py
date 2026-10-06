import csv

import pytest
import torch

from lottery import (
    CsvLogger,
    GlobalMagnitudePruning,
    LayerwiseMagnitudePruning,
    WinningTicket,
    load_checkpoint,
    save_checkpoint,
)

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
    assert payload["format_version"] == 2


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


@pytest.mark.parametrize(
    "strategy", [GlobalMagnitudePruning(), LayerwiseMagnitudePruning()], ids=["global", "layerwise"]
)
def test_resumed_search_prunes_the_same_weights_as_an_uninterrupted_one(
    tmp_path, trainer, strategy
):
    """Loading a checkpoint leaves the derived `weight` tensor stale until a forward pass."""
    original = WinningTicket(TinyNet(), trainer, strategy=strategy)
    original.search(rounds=1, epochs=1)
    path = original.save(tmp_path / "t.pt")
    original.search(rounds=1, epochs=1)

    resumed = WinningTicket(TinyNet(), trainer, strategy=strategy)
    resumed.load(path)
    resumed.search(rounds=1, epochs=1)

    for key, mask in original.masks().items():
        assert torch.equal(resumed.masks()[key], mask), key


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rewind": "none"}, "rewind='weights'"),
        ({"rewind_step": 2}, "rewind_step=0"),
    ],
)
def test_resuming_with_different_rewind_settings_is_rejected(tmp_path, kwargs, match):
    source = WinningTicket(TinyNet(), ShiftTrainer())
    source.search(rounds=1, epochs=1)
    path = source.save(tmp_path / "t.pt")
    with pytest.raises(ValueError, match=match):
        WinningTicket(TinyNet(), ShiftTrainer(), **kwargs).load(path)


def test_resuming_with_a_different_strategy_warns(tmp_path):
    source = WinningTicket(TinyNet(), ShiftTrainer())
    path = source.save(tmp_path / "t.pt")
    resumed = WinningTicket(TinyNet(), ShiftTrainer(), strategy=LayerwiseMagnitudePruning())
    with pytest.warns(UserWarning, match="GlobalMagnitudePruning"):
        resumed.load(path)


def test_random_rewind_resume_is_reproducible(tmp_path):
    source = WinningTicket(TinyNet(), ShiftTrainer(), rewind="random")
    source.search(rounds=1, epochs=1)
    path = source.save(tmp_path / "t.pt")
    source.search(rounds=1, epochs=1)

    torch.manual_seed(1234)  # whatever the RNG was doing in between
    resumed = WinningTicket(TinyNet(), ShiftTrainer(), rewind="random")
    resumed.load(path)
    resumed.search(rounds=1, epochs=1)
    for key, value in source.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[key], value), key


def test_checkpoint_without_history_cannot_resume(tmp_path):
    model = WinningTicket(TinyNet(), ShiftTrainer(), rewind="none").model
    path = save_checkpoint(tmp_path / "c.pt", model=model, rewind_state=None, rounds_completed=2)
    with pytest.raises(ValueError, match="history"):
        WinningTicket(TinyNet(), ShiftTrainer(), rewind="none").load(path)


def test_version_1_checkpoints_still_load(tmp_path):
    ticket = WinningTicket(TinyNet(), ShiftTrainer())
    ticket.search(rounds=1, epochs=1)
    path = ticket.save(tmp_path / "t.pt")
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 1
    del payload["config"], payload["rng_state"]
    torch.save(payload, path)

    resumed = WinningTicket(TinyNet(), ShiftTrainer())
    resumed.load(path)
    assert resumed.rounds_completed == 2


def _fake_gpus(monkeypatch, count: int, restored: list) -> None:
    states = [torch.full((4,), i, dtype=torch.uint8) for i in range(count)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: states)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.extend)


def test_cuda_rng_state_is_saved_and_restored(tmp_path, monkeypatch):
    restored: list = []
    _fake_gpus(monkeypatch, 2, restored)
    path = WinningTicket(TinyNet(), ShiftTrainer()).save(tmp_path / "t.pt")
    assert len(torch.load(path, weights_only=True)["cuda_rng_state"]) == 2

    WinningTicket(TinyNet(), ShiftTrainer()).load(path)
    assert [int(s[0]) for s in restored] == [0, 1]


def test_cuda_rng_state_from_a_different_gpu_count_is_not_restored(tmp_path, monkeypatch):
    restored: list = []
    _fake_gpus(monkeypatch, 2, restored)
    path = WinningTicket(TinyNet(), ShiftTrainer()).save(tmp_path / "t.pt")

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.warns(UserWarning, match="2 GPUs"):
        WinningTicket(TinyNet(), ShiftTrainer()).load(path)
    assert restored == []


def test_no_cuda_rng_state_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = WinningTicket(TinyNet(), ShiftTrainer()).save(tmp_path / "t.pt")
    assert torch.load(path, weights_only=True)["cuda_rng_state"] is None


def test_a_rejected_checkpoint_leaves_the_ticket_untouched(tmp_path):
    source = WinningTicket(TinyNet(), ShiftTrainer())
    source.search(rounds=2, epochs=1)
    path = source.save(tmp_path / "t.pt")

    ticket = WinningTicket(TinyNet(), ShiftTrainer(), rewind="none")
    before = {k: v.clone() for k, v in ticket.model.state_dict().items()}
    with pytest.raises(ValueError, match="rewind"):
        ticket.load(path)
    assert ticket.rounds_completed == 0
    for key, value in ticket.model.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_resuming_with_different_strategy_settings_warns(tmp_path):
    source = WinningTicket(
        TinyNet(), ShiftTrainer(), strategy=LayerwiseMagnitudePruning(output_layer_scale=0.5)
    )
    path = source.save(tmp_path / "t.pt")
    resumed = WinningTicket(TinyNet(), ShiftTrainer(), strategy=LayerwiseMagnitudePruning())
    with pytest.warns(UserWarning, match="output_layer_scale=0.5"):
        resumed.load(path)
