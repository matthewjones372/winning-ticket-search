import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lottery.training import (
    ClassificationTrainer,
    Metrics,
    _Accumulator,
    adam,
    cosine_annealing,
    sgd,
)


def test_reported_loss_is_the_actual_loss(tiny_net, trainer, classification_data):
    """Regression: 'loss' used to be MAE between predicted and true class indices."""
    tiny_net.eval()
    with torch.no_grad():
        xs, ys = zip(*classification_data, strict=True)
        logits = tiny_net(torch.cat(xs))
        labels = torch.cat(ys)
        expected_loss = nn.functional.cross_entropy(logits, labels).item()
        expected_acc = (logits.argmax(1) == labels).float().mean().item()

    metrics = trainer.evaluate(tiny_net)

    assert metrics.loss == pytest.approx(expected_loss, rel=1e-5)
    assert metrics.accuracy == pytest.approx(expected_acc)


def test_loss_is_sample_weighted_across_uneven_batches():
    acc = _Accumulator()
    targets_a, targets_b = torch.zeros(3, dtype=torch.long), torch.zeros(1, dtype=torch.long)
    acc.update(torch.tensor(1.0), torch.tensor([[1.0, 0.0]] * 3), targets_a)
    acc.update(torch.tensor(5.0), torch.tensor([[0.0, 1.0]]), targets_b)
    assert acc.result() == Metrics(loss=2.0, accuracy=0.75)


def test_empty_loader_gives_nan_metrics():
    result = _Accumulator().result()
    assert math.isnan(result.loss)
    assert math.isnan(result.accuracy)


def test_fit_always_returns_per_epoch_results(tiny_net, trainer):
    """Regression: results were only collected when logging was switched on."""
    results = trainer.fit(tiny_net, epochs=3)
    assert [r.epoch for r in results] == [0, 1, 2]
    assert all(0.0 <= r.test.accuracy <= 1.0 for r in results)


def test_fit_learns_a_separable_problem(classification_data):
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 3))
    trainer = ClassificationTrainer(
        nn.CrossEntropyLoss(), classification_data, classification_data, optimiser=adam(lr=1e-2)
    )
    results = trainer.fit(model, epochs=30)
    assert results[-1].train.loss < results[0].train.loss
    assert results[-1].test.accuracy > 0.85


def test_on_step_called_with_global_step(tiny_net, trainer):
    steps: list[int] = []
    trainer.fit(tiny_net, epochs=2, on_step=lambda step, model: steps.append(step))
    assert steps == list(range(1, 7))  # 3 batches x 2 epochs


def test_scheduler_is_built_per_fit_and_stepped_each_epoch(tiny_net, classification_data):
    built = []

    def scheduler(optimiser, epochs):
        sched = cosine_annealing(optimiser, epochs)
        built.append((optimiser, epochs))
        return sched

    trainer = ClassificationTrainer(
        nn.CrossEntropyLoss(),
        classification_data,
        classification_data,
        optimiser=sgd(lr=0.1),
        scheduler=scheduler,
    )
    trainer.fit(tiny_net, epochs=4)
    trainer.fit(tiny_net, epochs=2)

    assert [epochs for _, epochs in built] == [4, 2]
    assert built[0][0] is not built[1][0], "each round must get a fresh optimiser"
    # cosine annealing over T_max epochs ends at eta_min=0 after T_max steps
    assert built[0][0].param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)


def test_bf16_autocast_on_cpu(tiny_net, classification_data):
    trainer = ClassificationTrainer(
        nn.CrossEntropyLoss(),
        classification_data,
        classification_data,
        autocast_dtype=torch.bfloat16,
    )
    results = trainer.fit(tiny_net, epochs=1)
    assert tiny_net.fc1.weight.dtype == torch.float32  # master weights stay fp32
    assert math.isfinite(results[0].train.loss)


def test_optimiser_factories():
    params = [nn.Parameter(torch.zeros(1))]
    assert isinstance(sgd()(params), torch.optim.SGD)
    opt = adam(lr=0.5, weight_decay=0.1)(params)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == 0.5


def test_frozen_parameters_are_not_optimised(classification_data):
    model = nn.Sequential(nn.Linear(8, 3))
    model[0].bias.requires_grad_(False)
    before = model[0].bias.clone()
    loader = DataLoader(TensorDataset(torch.randn(8, 8), torch.zeros(8, dtype=torch.long)))
    ClassificationTrainer(nn.CrossEntropyLoss(), loader, loader).fit(model, epochs=1)
    assert torch.equal(model[0].bias, before)


def test_val_loader_is_evaluated_every_epoch(tiny_net, classification_data):
    trainer = ClassificationTrainer(
        nn.CrossEntropyLoss(),
        classification_data,
        classification_data,
        val_loader=classification_data,
    )
    results = trainer.fit(tiny_net, epochs=2)
    assert all(r.val is not None for r in results)
    assert results[-1].val == results[-1].test, "same data, same numbers"


def test_no_val_loader_means_no_val_metrics(tiny_net, trainer):
    assert trainer.fit(tiny_net, epochs=1)[0].val is None


def test_on_epoch_called_with_each_result(tiny_net, trainer):
    seen = []
    results = trainer.fit(tiny_net, epochs=3, on_epoch=seen.append)
    assert seen == results
