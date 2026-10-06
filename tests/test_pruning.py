import pytest
import torch
from torch import nn

from lottery.pruning import (
    GlobalMagnitudePruning,
    LayerSparsity,
    LayerwiseMagnitudePruning,
    attach_masks,
    default_prunable_parameters,
    get_mask,
    masked_parameters,
    overall_density,
    remove_masks,
    sparsity_report,
)


def _two_layers(small_scale: float = 0.01) -> nn.Sequential:
    model = nn.Sequential(nn.Linear(10, 10, bias=False), nn.Linear(10, 10, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.rand(10, 10) * small_scale + small_scale)  # tiny, positive
        model[1].weight.copy_(torch.rand(10, 10) + 1.0)  # large, positive
    return model


def test_default_selection_takes_linear_and_conv_weights_only():
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Flatten(), nn.LayerNorm(4), nn.Linear(4, 2)
    )
    selected = default_prunable_parameters(model)
    assert [(type(m), n) for m, n in selected] == [(nn.Conv2d, "weight"), (nn.Linear, "weight")]


def test_global_pruning_ranks_across_layers():
    """Regression: the old 'global' strategy took a percentile inside each layer."""
    model = _two_layers()
    params = default_prunable_parameters(model)
    attach_masks(params)

    GlobalMagnitudePruning().prune(params, 0.5)

    first, second = (get_mask(m, n) for m, n in params)
    assert first.sum() == 0, "every weight in the small-magnitude layer should be pruned"
    assert second.sum() == 100


def test_layerwise_pruning_prunes_each_layer_equally():
    model = _two_layers()
    params = default_prunable_parameters(model)
    attach_masks(params)

    LayerwiseMagnitudePruning().prune(params, 0.3)

    assert [int(get_mask(m, n).sum()) for m, n in params] == [70, 70]


def test_layerwise_output_layer_scale():
    model = _two_layers()
    params = default_prunable_parameters(model)

    LayerwiseMagnitudePruning(output_layer_scale=0.5).prune(params, 0.4)

    assert [int(get_mask(m, n).sum()) for m, n in params] == [60, 80]


def test_layerwise_output_layer_scale_zero_skips_last_layer():
    model = _two_layers()
    params = default_prunable_parameters(model)
    attach_masks(params)

    LayerwiseMagnitudePruning(output_layer_scale=0.0).prune(params, 0.4)

    assert [int(get_mask(m, n).sum()) for m, n in params] == [60, 100]


@pytest.mark.parametrize("strategy", [GlobalMagnitudePruning(), LayerwiseMagnitudePruning()])
def test_repeated_pruning_takes_a_fraction_of_the_remaining_weights(strategy):
    model = nn.Sequential(nn.Linear(50, 40), nn.Linear(40, 25))
    params = default_prunable_parameters(model)
    attach_masks(params)
    total = sum(m.weight.numel() for m, _ in params)

    for k in range(1, 4):
        strategy.prune(params, 0.2)
        remaining = sum(int(get_mask(m, n).sum()) for m, n in params)
        assert remaining == pytest.approx(total * 0.8**k, abs=2)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
@pytest.mark.parametrize("strategy", [GlobalMagnitudePruning(), LayerwiseMagnitudePruning()])
def test_invalid_fraction_rejected(strategy, fraction):
    params = default_prunable_parameters(nn.Linear(3, 3))
    with pytest.raises(ValueError, match="fraction"):
        strategy.prune(params, fraction)


def test_invalid_output_layer_scale_rejected():
    params = default_prunable_parameters(nn.Linear(3, 3))
    with pytest.raises(ValueError, match="output_layer_scale"):
        LayerwiseMagnitudePruning(output_layer_scale=2.0).prune(params, 0.5)


def test_pruned_weights_receive_no_gradient_and_negative_weights_still_learn():
    """Regression: the old gradient masking used `p < eps`, freezing all negative weights."""
    layer = nn.Linear(20, 20)
    params = default_prunable_parameters(layer)
    GlobalMagnitudePruning().prune(params, 0.5)

    layer(torch.randn(8, 20)).pow(2).sum().backward()

    grad, mask, orig = layer.weight_orig.grad, layer.weight_mask, layer.weight_orig
    assert torch.all(grad[mask == 0] == 0)
    alive_negative = (mask == 1) & (orig < 0)
    assert alive_negative.any()
    assert torch.all(grad[alive_negative] != 0)


def test_attach_masks_is_idempotent():
    layer = nn.Linear(4, 4)
    params = default_prunable_parameters(layer)
    attach_masks(params)
    GlobalMagnitudePruning().prune(params, 0.5)
    attach_masks(params)  # must not reset the existing mask
    assert int(layer.weight_mask.sum()) == 8


def test_get_mask_on_unpruned_module_is_all_ones():
    layer = nn.Linear(3, 2)
    assert torch.equal(get_mask(layer, "weight"), torch.ones(2, 3))


def test_remove_masks_bakes_zeros_into_weights():
    layer = nn.Linear(10, 10)
    params = default_prunable_parameters(layer)
    GlobalMagnitudePruning().prune(params, 0.5)

    remove_masks(params)
    remove_masks(params)  # no-op the second time

    assert isinstance(layer.weight, nn.Parameter)
    assert not hasattr(layer, "weight_mask")
    assert int((layer.weight == 0).sum()) == 50


def test_sparsity_report_counts_each_layer_separately():
    """Regression: the old weight log wrote a running total against every layer name."""
    model = _two_layers()
    params = default_prunable_parameters(model)
    LayerwiseMagnitudePruning().prune(params, 0.5)
    LayerwiseMagnitudePruning(output_layer_scale=0.0).prune(params, 0.5)

    report = sparsity_report(model, params)

    assert report == [
        LayerSparsity(name="0.weight", remaining=25, total=100),
        LayerSparsity(name="1.weight", remaining=50, total=100),
    ]
    assert report[0].density == 0.25
    assert overall_density(report) == pytest.approx(0.375)


def test_density_of_empty_report():
    assert overall_density([]) == 0.0
    assert LayerSparsity("x", 0, 0).density == 0.0


def test_global_pruning_ranks_live_weights_not_the_stale_forward_copy():
    """`module.weight` is only refreshed by a forward pass; ranking must use weight_orig."""
    model = nn.Sequential(nn.Linear(10, 10, bias=False), nn.Linear(10, 10, bias=False))
    params = default_prunable_parameters(model)
    attach_masks(params)
    with torch.no_grad():  # simulate training that never runs another forward
        model[0].weight_orig.fill_(0.01)
        model[1].weight_orig.fill_(1.0)

    GlobalMagnitudePruning().prune(params, 0.5)

    assert int(model[0].weight_mask.sum()) == 0
    assert int(model[1].weight_mask.sum()) == 100


def test_masked_parameters_finds_every_mask():
    model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    assert masked_parameters(model) == []
    attach_masks([(model[1], "weight")])
    assert masked_parameters(model) == [(model[1], "weight")]


def test_layerwise_pruning_ranks_live_weights_not_the_stale_forward_copy():
    model = nn.Sequential(nn.Linear(10, 10, bias=False))
    params = default_prunable_parameters(model)
    attach_masks(params)
    with torch.no_grad():  # simulate training that never runs another forward
        model[0].weight_orig.copy_(torch.arange(100.0).reshape(10, 10))

    LayerwiseMagnitudePruning().prune(params, 0.5)

    assert torch.equal(model[0].weight_mask.flatten(), (torch.arange(100) >= 50).float())
