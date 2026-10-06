"""Magnitude pruning strategies built on :mod:`torch.nn.utils.prune`.

Masks live on the modules as ``<name>_mask`` buffers and are applied in the forward
pass (``weight = weight_orig * weight_mask``). Pruned weights therefore receive zero
gradient automatically, so there is no need to patch gradients during training.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.nn.utils import prune

type PrunableParameter = tuple[nn.Module, str]
"""A ``(module, parameter_name)`` pair, the unit :mod:`torch.nn.utils.prune` works on."""

PRUNABLE_TYPES: tuple[type[nn.Module], ...] = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
)


def default_prunable_parameters(model: nn.Module) -> list[PrunableParameter]:
    """The weights of every linear and convolutional layer, in definition order.

    Biases and normalisation layers are deliberately left out, matching Frankle & Carbin.
    Subclasses (for example torchao's ``FakeQuantizedLinear``) are included.
    """
    return [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, PRUNABLE_TYPES) and getattr(module, "weight", None) is not None
    ]


type ParameterSelector = Callable[[nn.Module], Sequence[PrunableParameter]]


def attach_masks(parameters: Sequence[PrunableParameter]) -> None:
    """Attach an all-ones mask to each parameter that does not have one yet."""
    for module, name in parameters:
        if not hasattr(module, f"{name}_mask"):
            prune.identity(module, name)  # type: ignore[no-untyped-call]


def remove_masks(parameters: Sequence[PrunableParameter]) -> None:
    """Bake masks into the weights and drop the pruning re-parametrisation."""
    for module, name in parameters:
        if hasattr(module, f"{name}_mask"):
            prune.remove(module, name)  # type: ignore[no-untyped-call]


def masked_parameters(model: nn.Module) -> list[PrunableParameter]:
    """Every ``(module, name)`` in ``model`` that currently carries a pruning mask."""
    return [
        (module, buffer[: -len("_mask")])
        for module in model.modules()
        for buffer, _ in module.named_buffers(recurse=False)
        if buffer.endswith("_mask")
    ]


def detach_masked_weights(parameters: Sequence[PrunableParameter]) -> None:
    """Recompute ``weight = weight_orig * weight_mask`` outside autograd.

    After a forward pass the derived ``weight`` attribute is a non-leaf tensor, which
    makes ``copy.deepcopy`` of the model fail. Call this first to make it copyable.
    """
    with torch.no_grad():
        for module, name in parameters:
            mask = getattr(module, f"{name}_mask", None)
            if mask is not None:
                setattr(module, name, getattr(module, f"{name}_orig") * mask)


def get_mask(module: nn.Module, name: str) -> torch.Tensor:
    mask: torch.Tensor | None = getattr(module, f"{name}_mask", None)
    if mask is None:
        return torch.ones_like(getattr(module, name))
    return mask


class PruningStrategy(Protocol):
    """Prunes a fraction of the *remaining* weights of the given parameters in place."""

    def prune(self, parameters: Sequence[PrunableParameter], fraction: float) -> None: ...


def _check_fraction(fraction: float) -> None:
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"prune fraction must be in (0, 1), got {fraction}")


@dataclass(frozen=True, slots=True)
class GlobalMagnitudePruning:
    """Remove the smallest-magnitude weights ranked across *all* layers together."""

    def prune(self, parameters: Sequence[PrunableParameter], fraction: float) -> None:
        _check_fraction(fraction)
        scores = {(module, name): _live_weights(module, name) for module, name in parameters}
        prune.global_unstructured(
            list(parameters),
            pruning_method=prune.L1Unstructured,
            importance_scores=scores,
            amount=fraction,
        )


@dataclass(frozen=True, slots=True)
class LayerwiseMagnitudePruning:
    """Remove the smallest-magnitude weights within each layer independently.

    ``output_layer_scale`` multiplies the fraction for the final parameter. Frankle &
    Carbin prune the output layer of their fully-connected networks at half the rate
    (``output_layer_scale=0.5``).
    """

    output_layer_scale: float = 1.0

    def prune(self, parameters: Sequence[PrunableParameter], fraction: float) -> None:
        _check_fraction(fraction)
        if not 0.0 <= self.output_layer_scale <= 1.0:
            raise ValueError("output_layer_scale must be in [0, 1]")
        last = len(parameters) - 1
        for index, (module, name) in enumerate(parameters):
            amount = fraction * self.output_layer_scale if index == last else fraction
            if amount > 0:
                prune.l1_unstructured(  # type: ignore[no-untyped-call]
                    module, name, amount=amount, importance_scores=_live_weights(module, name)
                )


def _live_weights(module: nn.Module, name: str) -> torch.Tensor:
    """The masked weights as they are now, to rank by magnitude.

    The derived ``module.<name>`` tensor is only refreshed by a forward pass, so it is
    stale straight after training or after loading a checkpoint.
    """
    weights: torch.Tensor = getattr(module, f"{name}_orig", getattr(module, name))
    return weights.detach() * get_mask(module, name)


@dataclass(frozen=True, slots=True)
class LayerSparsity:
    name: str
    remaining: int
    total: int

    @property
    def density(self) -> float:
        return self.remaining / self.total if self.total else 0.0


def sparsity_report(
    model: nn.Module, parameters: Sequence[PrunableParameter]
) -> list[LayerSparsity]:
    """Per-layer count of weights still alive according to the masks."""
    names = {id(module): name for name, module in model.named_modules()}
    report = []
    for module, param_name in parameters:
        mask = get_mask(module, param_name)
        report.append(
            LayerSparsity(
                name=f"{names.get(id(module), type(module).__name__)}.{param_name}",
                remaining=int(mask.count_nonzero().item()),
                total=mask.numel(),
            )
        )
    return report


def overall_density(report: Sequence[LayerSparsity]) -> float:
    total = sum(layer.total for layer in report)
    return sum(layer.remaining for layer in report) / total if total else 0.0
