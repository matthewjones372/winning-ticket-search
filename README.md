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
- a silent library: standard `logging`, plus opt-in callbacks for progress bars and CSV files

## Quick start

```python
from torch import nn

from lottery import ClassificationTrainer, WinningTicket
from lottery.models import LeNet300100

trainer = ClassificationTrainer(
    loss_fn=nn.CrossEntropyLoss(),
    train_loader=train_loader,
    test_loader=test_loader,
    val_loader=val_loader,  # optional: picks the ticket without peeking at the test set
    device=device,
)

ticket = WinningTicket(LeNet300100(), trainer)

# Round 0 trains the dense network, then each round prunes 20% of the surviving
# weights, rewinds the survivors to their initial values and retrains.
result = ticket.search(rounds=10, epochs=5, prune_fraction=0.2)

# or keep going until at most 5% of the weights remain
result = ticket.search_to_density(0.05, epochs=5)

best = result.best()  # sparsest round within 0.5 points of the dense validation accuracy
print(best.round, best.density, best.final_test)
result.best(tolerance=0.01)  # or score rounds your own way with metric=...
```

With a `val_loader`, every epoch records validation metrics too, and `best()` chooses by
validation accuracy, so the chosen round's test accuracy is an honest estimate. Without
one it falls back to test accuracy, which flatters the result. The examples hold out the
last twelfth of the training set, as Frankle & Carbin do for MNIST.

```python
```

### Pruning strategy

```python
from lottery import GlobalMagnitudePruning, LayerwiseMagnitudePruning

WinningTicket(model, trainer, strategy=GlobalMagnitudePruning())  # default
# The paper prunes the output layer of its FC nets at half rate:
WinningTicket(model, trainer, strategy=LayerwiseMagnitudePruning(output_layer_scale=0.5))
```

The output layer is the last one selected for pruning, which with the default selector is
the last one *defined*. If your model defines its head first, name it:
`LayerwiseMagnitudePruning(output_layer_scale=0.5, output_layer=model.head)`.

By default the weights of every `Linear` and `Conv*d` layer are pruned. Biases and
normalisation layers are left alone. Pass `parameters=` to pick your own.

### Rewinding

```python
WinningTicket(model, trainer)  # rewind to init (the original LTH procedure)
WinningTicket(model, trainer, rewind_step=500)  # late rewinding, needed for deeper conv nets
WinningTicket(model, trainer, rewind="random")  # random re-init with each layer's own init
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

The int8 model is evaluated on CPU, where the default config runs. Pass
`quantised_device="cuda"` for GPU schemes. To pick the ticket by its quantised accuracy:

```python
result.best(metric=lambda r: r.extra_metrics["quantised"].accuracy)
```

### Logging, progress and metrics

The search prints nothing on its own. It logs one `INFO` line per round to the
`lottery.ticket` logger and one `DEBUG` line per epoch to `lottery.training`, so you
see them once your application configures logging:

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
# round 3  density 51.20%  test acc 0.9712  quantised 0.9650
```

Progress bars and files are callbacks:

```python
from lottery import CsvLogger, ProgressBar

ticket = WinningTicket(model, trainer, callbacks=[ProgressBar(), CsvLogger("results/run")])
```

`ProgressBar` shows a bar over rounds with one over the current round's epochs, and
routes log lines above them while open. `CsvLogger` writes `metrics.csv` (per epoch,
including validation metrics when there are any), `layers.csv` (per-layer sparsity) and `extra.csv` (extra
per-round evaluations such as the int8 accuracy). For anything else, such as TensorBoard
or an experiment tracker, subclass `Callback`:

```python
from lottery import Callback


class MyTracker(Callback):
    def on_round_start(self, ticket, round_, epochs): ...

    def on_epoch_end(self, ticket, round_, result): ...  # result is an EpochResult

    def on_round_end(self, ticket, result): ...  # result is a RoundResult

    def on_search_end(self, ticket):  # also called if a round raises
        ...
```

`on_epoch_end` needs a trainer whose `fit` accepts an `on_epoch` callback, as
`ClassificationTrainer` does. With a custom trainer that doesn't, the other hooks still run.

### Checkpoints

```python
ticket = WinningTicket(model, trainer, checkpoint_dir="results/run/checkpoints", checkpoint_every=5)
ticket.save("ticket.pt")

resumed = WinningTicket(Model(), trainer)
resumed.load("ticket.pt")
resumed.search(rounds=5, epochs=5)  # carries on pruning
```

With `checkpoint_dir` set, the last round of every `search` call is saved too, and
`checkpoint_every=n` also saves every *n*th round along the way.
Checkpoints are plain `state_dict`s plus the round history, and load with
`torch.load(weights_only=True)`. They also record the rewind settings, which must match
on resume, the pruning strategy (a mismatch warns) and the CPU RNG state, which `load`
restores so a resumed `rewind="random"` search draws the same weights. A `CsvLogger`
pointed at the directory of a resumed run keeps the rows for the rounds already in the
checkpoint and drops any after it.

## Examples

```shell
uv run --group cpu --extra vision python examples/mnist_lenet.py --rounds 10 --epochs 5
uv run --group cpu --extra vision python examples/mnist_lenet.py --rewind random   # compare against the control
uv run --group cu130 --extra vision python examples/cifar10_conv.py --rewind-step 500
uv run --group cpu --extra vision --extra qat python examples/mnist_qat.py
```

Add `--fake-data --limit 120` to any of them for a quick run on random images, without
downloading a dataset.

A short MNIST run (10k training images, 2 epochs a round) reproduces the paper's
qualitative result: rewound tickets hold or improve accuracy as they get sparser, while
randomly re-initialised ones degrade.

| density | rewind to init | random re-init |
|--------:|---------------:|---------------:|
| 100%    | 0.917          | 0.918          |
| 51%     | 0.931          | 0.900          |
| 26%     | 0.931          | 0.888          |

## Development

The project uses [uv](https://docs.astral.sh/uv/). Choose a PyTorch build with a
dependency group: `cpu` (also right for Apple Silicon) or `cu130` (Linux and Windows).
They are not extras, so `pip install lottery` uses whatever torch you already have.

```shell
uv sync --group cpu --extra qat --extra vision
uv run pytest --cov
uv run ruff check . && uv run ruff format --check .
uv run mypy src examples
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
- `enable_logging=True` and `base_name` are replaced by `callbacks=[CsvLogger(dir)]`, and nothing is printed unless you configure `logging` or add `ProgressBar()`.

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
