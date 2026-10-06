import csv

import pytest
import torch
from torch import nn

from lottery import (
    CsvLogger,
    LayerwiseMagnitudePruning,
    Rewind,
    RoundResult,
    SearchResult,
    WinningTicket,
    rounds_for_density,
)
from lottery.pruning import get_mask
from lottery.training import EpochResult, Metrics

from .conftest import ShiftTrainer, TinyNet, tensor


def test_round_zero_trains_the_dense_network():
    trainer = ShiftTrainer()
    ticket = WinningTicket(TinyNet(), trainer)
    result = ticket.search(rounds=0, epochs=2)
    assert [r.round for r in result.rounds] == [0]
    assert result.rounds[0].density == 1.0
    assert trainer.fit_calls == [2]


def test_density_follows_the_prune_schedule():
    ticket = WinningTicket(TinyNet(), ShiftTrainer())
    result = ticket.search(rounds=3, epochs=1, prune_fraction=0.2)
    assert [round(r.density, 2) for r in result.rounds] == [1.0, 0.8, 0.64, 0.51]


def test_survivors_are_rewound_to_the_original_init():
    """Regression: the old search re-drew a random init each round instead of rewinding to θ0."""
    model = TinyNet()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    ticket = WinningTicket(model, ShiftTrainer(delta=1.0))

    ticket.search(rounds=1, epochs=1)  # train, prune, rewind, train
    # ShiftTrainer added 3.0 to everything during the round-1 training. Undo that to see
    # what the survivors were rewound to at the start of the round.
    for module, name in ticket.parameters:
        key = next(k for k, m in model.named_modules() if m is module)
        orig = getattr(module, f"{name}_orig") - 3.0
        mask = getattr(module, f"{name}_mask").bool()
        assert torch.allclose(orig[mask], initial[f"{key}.{name}"][mask], atol=1e-5)
    assert torch.allclose(model.fc1.bias - 3.0, initial["fc1.bias"], atol=1e-5)
    assert torch.allclose(model.bn.weight - 3.0, initial["bn.weight"], atol=1e-5)


def test_late_rewinding_snapshots_the_weights_at_rewind_step():
    model = TinyNet()
    initial = model.fc1.weight.detach().clone()
    ticket = WinningTicket(model, ShiftTrainer(delta=1.0), rewind_step=2)
    assert ticket.rewind_state() is None

    ticket.search(rounds=1, epochs=1)

    snapshot = ticket.rewind_state()
    assert snapshot is not None
    assert torch.allclose(snapshot["fc1.weight_orig"], initial + 2.0)
    assert not any(k.endswith("_mask") for k in snapshot)


def test_late_rewind_step_beyond_the_dense_round_is_an_error():
    ticket = WinningTicket(TinyNet(), ShiftTrainer(steps_per_epoch=1), rewind_step=5)
    with pytest.raises(ValueError, match="never reached"):
        ticket.search(rounds=1, epochs=2)


def test_random_rewind_redraws_weights():
    model = TinyNet()
    initial = model.fc1.weight.detach().clone()
    ticket = WinningTicket(model, ShiftTrainer(delta=0.0), rewind=Rewind.RANDOM)
    ticket.search(rounds=1, epochs=1)
    mask = get_mask(model.fc1, "weight").bool()
    assert not torch.allclose(tensor(model.fc1, "weight_orig")[mask], initial[mask])
    assert ticket.density() < 1.0, "masks must survive the re-initialisation"


def test_no_rewind_keeps_trained_weights():
    model = TinyNet()
    initial = model.fc1.weight.detach().clone()
    ticket = WinningTicket(model, ShiftTrainer(delta=1.0), rewind="none")
    ticket.search(rounds=1, epochs=1)
    # 3 steps in round 0 + 3 steps in round 1, never reset
    assert torch.allclose(tensor(model.fc1, "weight_orig"), initial + 6.0)


def test_search_continues_from_the_previous_call():
    trainer = ShiftTrainer()
    ticket = WinningTicket(TinyNet(), trainer)
    ticket.search(rounds=1, epochs=1)
    result = ticket.search(rounds=2, epochs=1)
    assert [r.round for r in result.rounds] == [0, 1, 2, 3]
    assert ticket.rounds_completed == 4
    assert ticket.density() == pytest.approx(0.8**3, abs=0.005)


@pytest.mark.parametrize(
    ("density", "fraction", "expected"),
    [(1.0, 0.2, 0), (0.8, 0.2, 1), (0.64, 0.2, 2), (0.05, 0.2, 14), (0.1, 0.5, 4)],
)
def test_rounds_for_density(density, fraction, expected):
    """The old formula approximated log(1 - p) by -p, overshooting the round count."""
    assert rounds_for_density(density, fraction) == expected


@pytest.mark.parametrize(("density", "fraction"), [(0.0, 0.2), (1.5, 0.2), (0.5, 0.0), (0.5, 1.0)])
def test_rounds_for_density_validation(density, fraction):
    with pytest.raises(ValueError, match="must be in"):
        rounds_for_density(density, fraction)


def test_search_to_density_reaches_target_with_uneven_strategy():
    """Layerwise pruning with a slower output layer shrinks density by less than (1 - p)."""
    ticket = WinningTicket(
        TinyNet(),
        ShiftTrainer(),
        strategy=LayerwiseMagnitudePruning(output_layer_scale=0.5),
    )
    ticket.search_to_density(0.3, epochs=1, prune_fraction=0.2)
    assert ticket.density() <= 0.3


def test_search_to_density_fails_when_no_progress():
    class NoOp:
        def prune(self, parameters, fraction):
            pass

    ticket = WinningTicket(TinyNet(), ShiftTrainer(), strategy=NoOp())
    with pytest.raises(RuntimeError, match="no weights"):
        ticket.search_to_density(0.5, epochs=1)


def test_search_to_density_stops_at_target():
    ticket = WinningTicket(TinyNet(), ShiftTrainer())
    ticket.search_to_density(0.5, epochs=1, prune_fraction=0.2)
    assert ticket.density() <= 0.5
    assert ticket.rounds_completed == 1 + 4  # dense + ceil(log 0.5 / log 0.8)

    ticket.search_to_density(0.3, epochs=1, prune_fraction=0.2)  # continues, no re-run
    assert ticket.rounds_completed == 1 + 6


def test_custom_strategy_is_used():
    ticket = WinningTicket(
        TinyNet(),
        ShiftTrainer(),
        strategy=LayerwiseMagnitudePruning(output_layer_scale=0.0),
    )
    ticket.search(rounds=1, epochs=1, prune_fraction=0.5)
    assert [layer.density for layer in ticket.sparsity()] == [0.5, 1.0]


def test_batchnorm_and_biases_are_never_pruned():
    ticket = WinningTicket(TinyNet(), ShiftTrainer())
    ticket.search(rounds=2, epochs=1, prune_fraction=0.5)
    names = [layer.name for layer in ticket.sparsity()]
    assert names == ["fc1.weight", "fc2.weight"]
    assert not hasattr(ticket.model.bn, "weight_mask")


def test_masks_returns_copies():
    ticket = WinningTicket(TinyNet(), ShiftTrainer())
    masks = ticket.masks()
    assert set(masks) == {"fc1.weight_mask", "fc2.weight_mask"}
    masks["fc1.weight_mask"].zero_()
    assert ticket.density() == 1.0


def test_call_and_repr(tiny_net):
    ticket = WinningTicket(tiny_net, ShiftTrainer())
    tiny_net.eval()
    assert ticket(torch.randn(2, 8)).shape == (2, 3)
    assert repr(ticket) == "WinningTicket(density=1.0000, rounds=0)"


def test_csv_output(tmp_path):
    ticket = WinningTicket(TinyNet(), ShiftTrainer(), callbacks=[CsvLogger(tmp_path)])
    ticket.search(rounds=1, epochs=2)
    ticket.search(rounds=1, epochs=2)  # appends, does not truncate

    with (tmp_path / "metrics.csv").open() as fh:
        metrics = list(csv.DictReader(fh))
    with (tmp_path / "layers.csv").open() as fh:
        layers = list(csv.DictReader(fh))

    assert [(r["round"], r["epoch"]) for r in metrics] == [
        (str(r), str(e)) for r in range(3) for e in range(2)
    ]
    assert len(layers) == 3 * 2
    assert [row["layer"] for row in layers[-2:]] == ["fc1.weight", "fc2.weight"]
    last = layers[-2:]
    alive = sum(int(row["remaining"]) for row in last) / sum(int(row["total"]) for row in last)
    assert alive == pytest.approx(0.64, abs=0.01)


def test_checkpoints_written_every_n_rounds(tmp_path):
    ticket = WinningTicket(
        TinyNet(), ShiftTrainer(), checkpoint_dir=tmp_path / "checkpoints", checkpoint_every=2
    )
    ticket.search(rounds=4, epochs=1)
    written = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
    assert written == ["round_000.pt", "round_002.pt", "round_004.pt"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rewind_step": -1}, "rewind_step"),
        ({"checkpoint_every": 1}, "checkpoint_dir"),
        ({"checkpoint_every": 0, "checkpoint_dir": "x"}, "checkpoint_every"),
        ({"rewind": "sideways"}, "sideways"),
        ({"rewind": "random", "rewind_step": 3}, "only applies"),
    ],
)
def test_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        WinningTicket(TinyNet(), ShiftTrainer(), **kwargs)


def test_model_without_prunable_parameters_rejected():
    with pytest.raises(ValueError, match="no prunable"):
        WinningTicket(nn.Sequential(nn.ReLU()), ShiftTrainer())


@pytest.mark.parametrize(("rounds", "epochs"), [(-1, 1), (1, 0)])
def test_search_validation(rounds, epochs):
    ticket = WinningTicket(TinyNet(), ShiftTrainer())
    with pytest.raises(ValueError, match="must be >="):
        ticket.search(rounds=rounds, epochs=epochs)


def _round(index: int, density: float, accuracy: float | None) -> RoundResult:
    epochs = (
        [] if accuracy is None else [EpochResult(0, Metrics(0.0, accuracy), Metrics(0.0, accuracy))]
    )
    return RoundResult(round=index, density=density, epochs=epochs, layers=[])


def test_best_round_is_sparsest_within_tolerance_of_dense():
    result = SearchResult(
        [_round(0, 1.0, 0.90), _round(1, 0.8, 0.91), _round(2, 0.64, 0.897), _round(3, 0.5, 0.80)]
    )
    assert result.best().round == 2
    assert result.rounds[1].final_test == Metrics(0.0, 0.91)


def test_best_edge_cases():
    with pytest.raises(ValueError, match="no rounds"):
        SearchResult([]).best()
    with pytest.raises(ValueError, match="dense round"):
        SearchResult([_round(3, 0.5, 0.9)]).best()
    with pytest.raises(ValueError, match="no score"):
        SearchResult([_round(0, 1.0, float("nan")), _round(1, 0.8, 0.5)]).best()
    with pytest.raises(ValueError, match="no score"):
        SearchResult([_round(0, 1.0, None)]).best()
    assert _round(0, 1.0, None).final_test is None


def test_best_tolerance_and_metric_are_configurable():
    rounds = [_round(0, 1.0, 90.0), _round(1, 0.8, 89.7), _round(2, 0.64, 89.0)]
    assert SearchResult(rounds).best().round == 0, "0.005 is a fraction, not percentage points"
    assert SearchResult(rounds).best(tolerance=0.5).round == 1

    quantised = [
        RoundResult(r.round, r.density, r.epochs, [], {"quantised": Metrics(0.0, q)})
        for r, q in zip(rounds, [0.9, 0.9, 0.5], strict=True)
    ]
    best = SearchResult(quantised).best(
        tolerance=0.1, metric=lambda r: r.extra_metrics["quantised"].accuracy
    )
    assert best.round == 1, "round 2 is within tolerance on test acc but not on int8 acc"


def test_end_to_end_with_real_training(trainer):
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 3))
    initial = tensor(model[0], "weight").detach().clone()
    ticket = WinningTicket(model, trainer)
    result = ticket.search(rounds=2, epochs=2, prune_fraction=0.5)

    assert len(result.rounds) == 3
    assert ticket.density() == pytest.approx(0.25, abs=0.01)
    assert all(r.final_test is not None for r in result.rounds)
    snapshot = ticket.rewind_state()
    assert snapshot is not None
    assert torch.equal(snapshot["0.weight_orig"], initial)


@pytest.mark.parametrize("fraction", [0.0, 1.0, 20])
def test_bad_prune_fraction_rejected_before_any_training(fraction):
    trainer = ShiftTrainer()
    ticket = WinningTicket(TinyNet(), trainer)
    with pytest.raises(ValueError, match="prune_fraction"):
        ticket.search(rounds=2, epochs=1, prune_fraction=fraction)
    assert trainer.fit_calls == []
    assert ticket.rounds_completed == 0


def test_unreachable_rewind_step_rejected_before_training(trainer, classification_data):
    def fit(*args, **kwargs):
        raise AssertionError("trained before checking rewind_step")

    trainer.fit = fit
    steps = 2 * len(classification_data)
    ticket = WinningTicket(TinyNet(), trainer, rewind_step=steps + 1)
    with pytest.raises(ValueError, match=f"only {steps} steps"):
        ticket.search(rounds=1, epochs=2)


def test_final_round_of_each_search_is_checkpointed(tmp_path):
    ticket = WinningTicket(TinyNet(), ShiftTrainer(), checkpoint_dir=tmp_path, checkpoint_every=2)
    ticket.search(rounds=3, epochs=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "round_000.pt",
        "round_002.pt",
        "round_003.pt",
    ]


def test_checkpoint_dir_alone_saves_the_end_of_each_search(tmp_path):
    ticket = WinningTicket(TinyNet(), ShiftTrainer(), checkpoint_dir=tmp_path)
    ticket.search(rounds=2, epochs=1)
    ticket.search_to_density(0.5, epochs=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["round_002.pt", "round_004.pt"]


def test_random_rewind_redraws_any_parameter_with_its_layers_own_init():
    model = TinyNet()
    bias_before = model.fc1.bias.detach().clone()
    ticket = WinningTicket(
        model,
        ShiftTrainer(delta=0.0),
        parameters=lambda m: [(m.get_submodule("fc1"), "weight"), (m.get_submodule("fc1"), "bias")],
        rewind="random",
    )
    ticket.search(rounds=1, epochs=1, prune_fraction=0.5)
    alive = get_mask(model.fc1, "bias").bool()
    bias = tensor(model.fc1, "bias_orig")
    assert not torch.allclose(bias[alive], bias_before[alive])
    bound = 1 / 8**0.5  # nn.Linear's default bias init for 8 inputs
    assert bias.abs().max() <= bound


def test_random_rewind_needs_reset_parameters():
    class Raw(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))

        def forward(self, x):
            return x @ self.weight

    model = nn.Sequential(Raw())
    with pytest.raises(ValueError, match="reset_parameters"):
        WinningTicket(
            model,
            ShiftTrainer(),
            parameters=lambda m: [(m.get_submodule("0"), "weight")],
            rewind="random",
        )


def test_masks_and_rewind_state_tell_pruning_masks_from_other_buffers():
    from .test_pruning import WithMaskBuffer

    ticket = WinningTicket(WithMaskBuffer(), ShiftTrainer())
    assert set(ticket.masks()) == {"fc.weight_mask"}
    snapshot = ticket.rewind_state()
    assert snapshot is not None
    assert "attention_mask" in snapshot, "a user buffer must be rewound like any other state"
    assert "fc.weight_mask" not in snapshot


def test_masks_of_a_bare_layer():
    ticket = WinningTicket(nn.Linear(4, 4), ShiftTrainer())
    assert set(ticket.masks()) == {"weight_mask"}


def _val_round(index: int, density: float, val: float, test: float) -> RoundResult:
    epoch = EpochResult(0, Metrics(0.0, test), Metrics(0.0, test), Metrics(0.0, val))
    return RoundResult(round=index, density=density, epochs=[epoch], layers=[])


def test_best_selects_on_validation_when_there_is_one():
    rounds = [
        _val_round(0, 1.0, val=0.90, test=0.90),
        _val_round(1, 0.8, val=0.90, test=0.80),
        _val_round(2, 0.64, val=0.70, test=0.95),
    ]
    assert rounds[1].final_val == Metrics(0.0, 0.90)
    assert SearchResult(rounds).best().round == 1, "round 2 only looks good on test"


def test_val_metrics_survive_a_checkpoint(tmp_path):
    ticket = WinningTicket(TinyNet(), ShiftTrainer(val_accuracy=0.75))
    ticket.search(rounds=1, epochs=1)
    path = ticket.save(tmp_path / "t.pt")
    resumed = WinningTicket(TinyNet(), ShiftTrainer(val_accuracy=0.75))
    resumed.load(path)
    assert resumed.history == ticket.history


def test_csv_has_validation_columns(tmp_path):
    ticket = WinningTicket(
        TinyNet(), ShiftTrainer(val_accuracy=0.75), callbacks=[CsvLogger(tmp_path)]
    )
    ticket.search(rounds=0, epochs=1)
    with (tmp_path / "metrics.csv").open() as fh:
        row = next(csv.DictReader(fh))
    assert row["val_accuracy"] == "0.75"
    assert row["val_loss"] == "0.0"


def test_ticket_options_match_the_constructor():
    import inspect

    from lottery.ticket import TicketOptions

    keyword_only = [
        name
        for name, p in inspect.signature(WinningTicket.__init__).parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    assert keyword_only == list(TicketOptions.__annotations__)


class Attention(nn.Module):
    """Parameters whose modules have no public reset_parameters(): MultiheadAttention's
    in-projection, and a raw positional embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
        self.pos = nn.Parameter(torch.zeros(4, 8))
        self.fc = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.pos
        return self.fc(self.attn(h, h, h)[0].mean(1))


def _fc_only(m: nn.Module) -> list[tuple[nn.Module, str]]:
    return [(m.get_submodule("fc"), "weight")]


def test_random_rewind_refuses_parameters_it_cannot_redraw():
    """Regression: they used to keep their trained values in the 'random' control."""
    with pytest.raises(ValueError, match="reinit"):
        WinningTicket(Attention(), ShiftTrainer(), parameters=_fc_only, rewind="random")


def test_random_rewind_with_reinit_redraws_every_parameter():
    model = Attention()
    trained_once = None

    class Spy(ShiftTrainer):
        def fit(self, model, epochs, on_step=None, on_epoch=None, start_step=0):
            nonlocal trained_once
            results = super().fit(model, epochs, on_step, on_epoch, start_step)
            if trained_once is None:
                trained_once = tensor(model.attn, "in_proj_weight").detach().clone()
            return results

    ticket = WinningTicket(
        model, Spy(delta=1.0), parameters=_fc_only, rewind="random", reinit=Attention
    )
    ticket.search(rounds=1, epochs=1)

    # The factory's pos is zeros, and each round trains it by +1 for 3 steps.
    assert torch.equal(tensor(model, "pos"), torch.full((4, 8), 3.0))
    assert trained_once is not None
    assert not torch.allclose(tensor(model.attn, "in_proj_weight"), trained_once + 3.0)
    assert ticket.density() < 1.0, "masks survive the redraw"


def test_reinit_must_match_the_architecture():
    ticket = WinningTicket(
        Attention(), ShiftTrainer(), parameters=_fc_only, rewind="random", reinit=TinyNet
    )
    with pytest.raises(ValueError, match="reinit"):
        ticket.search(rounds=1, epochs=1)


def test_reinit_only_applies_to_random_rewind():
    with pytest.raises(ValueError, match="reinit"):
        WinningTicket(TinyNet(), ShiftTrainer(), reinit=TinyNet)


def test_late_rewinding_resumes_training_at_the_rewind_step():
    """Frankle et al. 2020: rewind to W_k, then train the remaining T - k steps."""
    starts: list[int] = []

    class Spy(ShiftTrainer):
        def fit(self, model, epochs, on_step=None, on_epoch=None, start_step=0):
            starts.append(start_step)
            return super().fit(model, epochs, on_step, on_epoch, start_step)

    WinningTicket(TinyNet(), Spy(), rewind_step=2).search(rounds=2, epochs=2)
    assert starts == [0, 2, 2]


@pytest.mark.parametrize("rewind", ["weights", "none", "random"])
def test_other_rewinds_train_the_whole_schedule(rewind):
    starts: list[int] = []

    class Spy(ShiftTrainer):
        def fit(self, model, epochs, on_step=None, on_epoch=None, start_step=0):
            starts.append(start_step)
            return super().fit(model, epochs, on_step, on_epoch, start_step)

    WinningTicket(TinyNet(), Spy(), rewind=rewind).search(rounds=2, epochs=1)
    assert starts == [0, 0, 0]


def test_trainer_without_start_step_warns_once_about_late_rewinding():
    class OldTrainer:
        def __init__(self):
            self._inner = ShiftTrainer()

        def fit(self, model, epochs, on_step=None):
            return self._inner.fit(model, epochs, on_step)

        def evaluate(self, model, device=None):
            return self._inner.evaluate(model, device)

    ticket = WinningTicket(TinyNet(), OldTrainer(), rewind_step=2)
    with pytest.warns(UserWarning, match="start_step") as caught:
        ticket.search(rounds=2, epochs=1)
    assert len([w for w in caught if "start_step" in str(w.message)]) == 1
