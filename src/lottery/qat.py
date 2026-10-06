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
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from lottery.pruning import detach_masked_weights, masked_parameters, remove_masks
from lottery.ticket import WinningTicket
from lottery.training import Metrics, Trainer

try:
    from torchao.quantization import (
        Int8DynamicActivationIntxWeightConfig,
        PerAxis,
        quantize_,
    )
    from torchao.quantization.qat import QATConfig
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "lottery.qat needs torchao. Install the extra with `uv add 'lottery[qat]'`."
    ) from exc

if TYPE_CHECKING:
    from torchao.core.config import AOBaseConfig

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
    kwargs: dict[str, Any] = {} if filter_fn is None else {"filter_fn": filter_fn}
    quantize_(model, QATConfig(base_config or default_qat_config(), step="prepare"), **kwargs)
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
    kwargs: dict[str, Any] = {} if filter_fn is None else {"filter_fn": filter_fn}
    quantize_(converted, QATConfig(base_config or default_qat_config(), step="convert"), **kwargs)
    return converted.eval()


class QatWinningTicket(WinningTicket):
    """:class:`WinningTicket` for quantisation-aware training.

    The trainer trains and evaluates the fake-quantised model. When
    ``evaluate_quantised`` is set, each round also converts a copy to a real quantised
    model and records its test metrics under ``extra_metrics["quantised"]``.
    """

    def __init__(
        self,
        model: nn.Module,
        trainer: Trainer,
        *,
        base_config: AOBaseConfig | None = None,
        filter_fn: ModuleFilter | None = None,
        evaluate_quantised: bool = True,
        **kwargs: Any,
    ) -> None:
        self.base_config = base_config or default_qat_config()
        self.filter_fn = filter_fn
        self.evaluate_quantised = evaluate_quantised
        prepare_qat(model, self.base_config, filter_fn)
        super().__init__(model, trainer, **kwargs)

    def quantised_model(self) -> nn.Module:
        return convert_qat(self.model, self.base_config, self.filter_fn)

    def _round_metrics(self) -> dict[str, Metrics]:
        if not self.evaluate_quantised:
            return {}
        return {"quantised": self.trainer.evaluate(self.quantised_model())}
