"""Quantisation-aware training (QAT) on top of the lottery ticket search, using torchao.

Requires the ``qat`` extra: ``uv add 'lottery[qat]'``.

Linear layers are swapped for torchao's ``FakeQuantizedLinear`` *before* pruning masks
are attached, so training sees both the sparsity mask and simulated quantisation. At
conversion time the masks are baked into the weights first and the model is then
quantised for real. Pruned weights stay exactly zero under symmetric weight schemes
(the default); asymmetric ones such as int4 weight-only may map them to a small
non-zero value.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Unpack

import torch
from torch import nn

from lottery.pruning import detach_masked_weights, masked_parameters, remove_masks
from lottery.ticket import TicketOptions, WinningTicket

try:
    from torchao.quantization import Int8DynamicActivationIntxWeightConfig, PerAxis
    from torchao.quantization.qat import QATConfig, QATStep

    # Imported from where it is defined: `torchao.quantization` also has a `quantize_`
    # subpackage, which type checkers resolve instead of the re-exported function.
    from torchao.quantization.quant_api import quantize_
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "lottery.qat needs torchao. Install the extra with `uv add 'lottery[qat]'`."
    ) from exc

if TYPE_CHECKING:
    from torchao.core.config import AOBaseConfig

    from lottery.training import Metrics, Trainer

type ModuleFilter = Callable[[nn.Module, str], bool]


def default_qat_config() -> AOBaseConfig:
    """int8 dynamic per-token activations with int8 per-channel symmetric weights.

    Runs on CPU, so it is a sensible default for small research models. Pass any other
    base config torchao's ``QATConfig`` accepts (e.g. ``Int4WeightOnlyConfig``) for GPU.
    """
    return Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int8, weight_granularity=PerAxis(0)
    )


def prepare_qat(
    model: nn.Module,
    base_config: AOBaseConfig | None = None,
    filter_fn: ModuleFilter | None = None,
) -> nn.Module:
    """Insert fake quantisation into ``model`` in place (linear layers by default)."""
    _quantize(model, QATStep.PREPARE, base_config, filter_fn)
    return model


def convert_qat(
    model: nn.Module,
    base_config: AOBaseConfig | None = None,
    filter_fn: ModuleFilter | None = None,
) -> nn.Module:
    """Return a truly quantised copy of a fake-quantised (and possibly pruned) model."""
    detach_masked_weights(masked_parameters(model))
    converted = copy.deepcopy(model)
    remove_masks(masked_parameters(converted))
    _quantize(converted, QATStep.CONVERT, base_config, filter_fn)
    return converted.eval()


def _quantize(
    model: nn.Module,
    step: QATStep,
    base_config: AOBaseConfig | None,
    filter_fn: ModuleFilter | None,
) -> None:
    config = QATConfig(base_config or default_qat_config(), step=step)
    if filter_fn is None:
        quantize_(model, config)  # torchao's default filter: linear layers
    else:
        quantize_(model, config, filter_fn=filter_fn)


class QatWinningTicket(WinningTicket):
    """:class:`WinningTicket` for quantisation-aware training.

    The trainer trains and evaluates the fake-quantised model. When
    ``evaluate_quantised`` is set, each round also converts a copy to a real quantised
    model and records its test metrics under ``extra_metrics["quantised"]``, evaluated
    on ``quantised_device``. That defaults to CPU, where the default int8 config runs;
    set it to ``"cuda"`` for GPU schemes such as int4 weight-only.
    """

    def __init__(
        self,
        model: nn.Module,
        trainer: Trainer,
        *,
        base_config: AOBaseConfig | None = None,
        filter_fn: ModuleFilter | None = None,
        evaluate_quantised: bool = True,
        quantised_device: torch.device | str = "cpu",
        **options: Unpack[TicketOptions],
    ) -> None:
        self.base_config = base_config or default_qat_config()
        self.quantised_device = torch.device(quantised_device)
        self.filter_fn = filter_fn
        self.evaluate_quantised = evaluate_quantised
        prepare_qat(model, self.base_config, filter_fn)
        super().__init__(model, trainer, **options)

    def quantised_model(self) -> nn.Module:
        return convert_qat(self.model, self.base_config, self.filter_fn)

    def _round_metrics(self) -> dict[str, Metrics]:
        if not self.evaluate_quantised:
            return {}
        return {"quantised": self.trainer.evaluate(self.quantised_model(), self.quantised_device)}
