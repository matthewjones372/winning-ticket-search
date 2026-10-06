# Winning Ticket Search

[![CI](https://github.com/matthewjones372/winning-ticket-search/actions/workflows/ci.yml/badge.svg)](https://github.com/matthewjones372/winning-ticket-search/actions/workflows/ci.yml)

Iterative magnitude pruning (IMP) in PyTorch for finding lottery tickets, as described by
Frankle & Carbin in [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635).
Originally written for my MSc in Machine Learning.

It supports:

- global or layer-wise magnitude pruning, built on `torch.nn.utils.prune`
- rewinding to the original init, late rewinding to step *k* ([Frankle et al. 2020](https://arxiv.org/abs/1912.05671)), random re-init as a control, or no rewind (learning-rate rewinding, [Renda et al. 2020](https://arxiv.org/abs/2003.02389))
- quantisation-aware training through [torchao](https://github.com/pytorch/ao), with the real int8 model evaluated every round
- per-round CSV metrics, per-layer sparsity, and resumable `state_dict` checkpoints

## Quick start

```python
from torch import nn

from lottery import ClassificationTrainer, WinningTicket
from lottery.models import LeNet300100

trainer = ClassificationTrainer(
    loss_fn=nn.CrossEntropyLoss(),
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
)

ticket = WinningTicket(LeNet300100(), trainer, output_dir="results/lenet")

# Round 0 trains the dense network, then each round prunes 20% of the surviving
# weights, rewinds the survivors to their initial values and retrains.
result = ticket.search(rounds=10, epochs=5, prune_fraction=0.2)

# or keep going until at most 5% of the weights remain
result = ticket.search_to_density(0.05, epochs=5)

best = result.best()  # sparsest round within 0.5 points of the dense accuracy
print(best.round, best.density, best.final_test)
```

### Pruning strategy

```python
from lottery import GlobalMagnitudePruning, LayerwiseMagnitudePruning

WinningTicket(model, trainer, strategy=GlobalMagnitudePruning())  # default
# The paper prunes the output layer of its FC nets at half rate:
WinningTicket(model, trainer, strategy=LayerwiseMagnitudePruning(output_layer_scale=0.5))
```

By default the weights of every `Linear` and `Conv*d` layer are pruned. Biases and
normalisation layers are left alone. Pass `parameters=` to pick your own.

### Rewinding

```python
WinningTicket(model, trainer)  # rewind to init (the original LTH procedure)
WinningTicket(model, trainer, rewind_step=500)  # late rewinding, needed for deeper conv nets
WinningTicket(model, trainer, rewind="random")  # random re-init control
WinningTicket(model, trainer, rewind="none")  # keep trained weights, restart the LR schedule
```

### Quantisation-aware training

```python
from lottery.qat import QatWinningTicket

ticket = QatWinningTicket(model, trainer)  # int8 weights + activations by default, runs on CPU
result = ticket.search(rounds=5, epochs=3)
result.rounds[-1].extra_metrics["quantised"]  # accuracy of the real int8 model
int8_model = ticket.quantised_model()
```

Any base config torchao's `QATConfig` accepts can be passed as `base_config=`, for
example `Int4WeightOnlyConfig` for GPU inference. Symmetric weight schemes (the default)
keep pruned weights exactly zero after conversion; asymmetric ones may not.

### Checkpoints

```python
ticket = WinningTicket(model, trainer, output_dir="results/run", checkpoint_every=5)
ticket.save("ticket.pt")

resumed = WinningTicket(Model(), trainer)
resumed.load("ticket.pt")
resumed.search(rounds=5, epochs=5)  # carries on pruning
```

Checkpoints are plain `state_dict`s plus the round history, and load with
`torch.load(weights_only=True)`. Resuming into the same `output_dir` keeps the CSV rows
for the rounds already in the checkpoint.

## Examples

```shell
uv run --extra cpu python examples/mnist_lenet.py --rounds 10 --epochs 5
uv run --extra cpu python examples/mnist_lenet.py --rewind random   # compare against the control
uv run --extra cu130 python examples/cifar10_conv.py --rewind-step 500
uv run --extra cpu --extra qat python examples/mnist_qat.py
```

A short MNIST run (10k training images, 2 epochs a round) reproduces the paper's
qualitative result: rewound tickets hold or improve accuracy as they get sparser, while
randomly re-initialised ones degrade.

| density | rewind to init | random re-init |
|--------:|---------------:|---------------:|
| 100%    | 0.917          | 0.918          |
| 51%     | 0.931          | 0.900          |
| 26%     | 0.931          | 0.888          |

## Development

The project uses [uv](https://docs.astral.sh/uv/). Choose a PyTorch build with an extra:
`cpu` (also right for Apple Silicon) or `cu130`.

```shell
uv sync --extra cpu --extra qat --extra vision
uv run pytest --cov
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

## Upgrading from 1.x

2.0 is a rewrite with a new API:

- `WinningTicket(model, model_trainer, device, ...)` is now `WinningTicket(model, trainer, ...)`. The device lives on the trainer.
- `percentage_prune=5` (a percentage) is now `prune_fraction=0.05`.
- `search(prune_iterations, training_iterations)` is now `search(rounds, epochs)`.
- `search_by_target_sparsity(target_sparsity=5)` is now `search_to_density(0.05, ...)`.
- `ClassificationTrainer(loss_func=..., optimiser_type=OptimiserType.SGD)` is now `ClassificationTrainer(loss_fn=..., optimiser=sgd())`.
- `WinningQatTicket` is now `lottery.qat.QatWinningTicket`, built on torchao instead of `torch.quantization`. Models no longer need `QuantStub` and `DeQuantStub`.
- TorchScript checkpoints are replaced by `state_dict` checkpoints.

1.x had several correctness bugs that this release fixes:

- after round 0, survivors were randomly re-initialised instead of rewound to their initial values, so the search ran the random re-init control rather than finding tickets
- gradient masking zeroed the gradient of every negative weight, not only pruned ones
- the reported "loss" was the mean absolute error between predicted and true class indices
- per-epoch results were dropped when logging was turned off
- "global" pruning took a percentile per layer, and also pruned BatchNorm weights
- the weights CSV logged a running total against each layer name
- `search_by_target_sparsity` approximated `log(1 - p)` by `-p`, which can give the wrong number of rounds

## License

MIT
