import copy

import pytest
import torch
from torch import nn

pytest.importorskip("torchao")

from torchao.quantization import IntxUnpackedToInt8Tensor
from torchao.quantization.qat import FakeQuantizedLinear

from lottery import CsvLogger
from lottery.pruning import GlobalMagnitudePruning, default_prunable_parameters, get_mask
from lottery.qat import QatWinningTicket, convert_qat, prepare_qat

from .conftest import ShiftTrainer, TinyNet, tensor


def test_prepare_swaps_linear_layers_for_fake_quantised_ones():
    model = prepare_qat(TinyNet())
    assert isinstance(model.get_submodule("fc1"), FakeQuantizedLinear)
    assert isinstance(model.bn, nn.BatchNorm1d)


def test_prepare_respects_filter():
    model = prepare_qat(TinyNet(), filter_fn=lambda module, fqn: fqn == "fc2")
    assert type(model.get_submodule("fc1")) is nn.Linear
    assert isinstance(model.get_submodule("fc2"), FakeQuantizedLinear)


def test_fake_quantised_layers_are_prunable_and_masked_in_forward():
    model = prepare_qat(TinyNet())
    params = default_prunable_parameters(model)
    assert len(params) == 2
    GlobalMagnitudePruning().prune(params, 0.5)
    model.eval()
    model(torch.randn(4, 8)).sum().backward()
    grad = tensor(model.get_submodule("fc1"), "weight_orig").grad
    assert grad is not None
    assert torch.all(grad[get_mask(model.get_submodule("fc1"), "weight") == 0] == 0)


def test_convert_preserves_sparsity_and_leaves_source_untouched():
    model = prepare_qat(TinyNet())
    params = default_prunable_parameters(model)
    GlobalMagnitudePruning().prune(params, 0.5)
    model.eval()

    quantised = convert_qat(model)

    weight = tensor(quantised.get_submodule("fc1"), "weight")
    assert type(quantised.get_submodule("fc1")) is nn.Linear
    assert isinstance(weight, IntxUnpackedToInt8Tensor)
    assert torch.all(weight.qdata[get_mask(model.get_submodule("fc1"), "weight") == 0] == 0)
    assert hasattr(model.get_submodule("fc1"), "weight_mask"), "source model must keep its masks"
    x = torch.randn(4, 8)
    assert torch.allclose(quantised(x), model(x), atol=0.2)


def test_qat_ticket_records_quantised_metrics():
    ticket = QatWinningTicket(TinyNet(), ShiftTrainer(delta=0.0))
    result = ticket.search(rounds=1, epochs=1)
    assert all("quantised" in r.extra_metrics for r in result.rounds)
    assert result.rounds[-1].density == pytest.approx(0.8, abs=0.01)


def test_quantised_model_is_evaluated_on_the_quantised_device():
    devices = []

    class Spy(ShiftTrainer):
        def evaluate(self, model, device=None):
            devices.append(device)
            return super().evaluate(model, device)

    QatWinningTicket(TinyNet(), Spy(delta=0.0)).search(rounds=0, epochs=1)
    QatWinningTicket(TinyNet(), Spy(delta=0.0), quantised_device="meta").search(rounds=0, epochs=1)
    assert devices == [torch.device("cpu"), torch.device("meta")]


def test_qat_ticket_can_skip_quantised_eval():
    ticket = QatWinningTicket(TinyNet(), ShiftTrainer(delta=0.0), evaluate_quantised=False)
    result = ticket.search(rounds=0, epochs=1)
    assert result.rounds[0].extra_metrics == {}


def test_qat_ticket_with_real_trainer(trainer, tmp_path):
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 3))
    ticket = QatWinningTicket(model, trainer, callbacks=[CsvLogger(tmp_path)])
    result = ticket.search(rounds=1, epochs=2)
    quantised = result.rounds[-1].extra_metrics["quantised"]
    fake = result.rounds[-1].final_test
    assert fake is not None
    assert abs(quantised.accuracy - fake.accuracy) < 0.1
    assert (tmp_path / "extra.csv").read_text().count("quantised") == 2
    assert copy.deepcopy(ticket.quantised_model()) is not None


def test_quantised_model_with_custom_parameter_selector():
    ticket = QatWinningTicket(
        TinyNet(),
        ShiftTrainer(delta=0.0),
        parameters=lambda m: [(m.get_submodule("fc2"), "weight")],
    )
    ticket.search(rounds=1, epochs=1, prune_fraction=0.5)
    quantised = ticket.quantised_model()
    weight = tensor(quantised.get_submodule("fc2"), "weight")
    assert isinstance(weight, IntxUnpackedToInt8Tensor)
    assert int((weight.qdata == 0).sum()) >= 24
