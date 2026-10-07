# Winning Ticket Search

[![CI](https://github.com/matthewjones372/winning-ticket-search/actions/workflows/ci.yml/badge.svg)](https://github.com/matthewjones372/winning-ticket-search/actions/workflows/ci.yml)

Iterative magnitude pruning (IMP) in PyTorch for finding lottery tickets, as described by
Frankle & Carbin in [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635).
Originally written for my MSc in Machine Learning; the report, *Enabling Deep Learning at the Edge*,
is in [`report-public/report.pdf`](report-public/report.pdf).

It supports:

- global or layer-wise magnitude pruning, built on `torch.nn.utils.prune`
- rewinding to the original init, late rewinding to step *k* ([Frankle et al. 2020](https://arxiv.org/abs/1912.05671)), or no rewind (learning-rate rewinding, [Renda et al. 2020](https://arxiv.org/abs/2003.02389))
- the random re-initialisation control: a winning ticket's masks trained from a fresh init
- quantisation-aware training through [torchao](https://github.com/pytorch/ao), with the real int8 model evaluated every round
- per-round CSV metrics, per-layer sparsity, and resumable `state_dict` checkpoints
- no output by default: standard `logging`, plus opt-in callbacks for progress bars and CSV files

## Install

The package is not on PyPI; install it from GitHub, pinned to a release:

```shell
uv add "lottery @ git+https://github.com/matthewjones372/winning-ticket-search@2.3.0"
uv add "lottery[qat] @ git+https://github.com/matthewjones372/winning-ticket-search@2.3.0"  # with quantisation-aware training via torchao
```

Bring your own PyTorch build; with pip, `pip install "lottery @ git+https://github.com/matthewjones372/winning-ticket-search@2.3.0"`.

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

### Pruning strategy

```python
from lottery import GlobalMagnitudePruning, LayerwiseMagnitudePruning

WinningTicket(model, trainer, strategy=GlobalMagnitudePruning())  # default
# The paper prunes the output layer of its FC nets at half rate:
WinningTicket(model, trainer, strategy=LayerwiseMagnitudePruning(output_layer_scale=0.5))
# and, in its Conv-2/4 nets, conv layers at 10% a round, dense 20%, output 10%:
strategy = LayerwiseMagnitudePruning(scales={nn.Conv2d: 0.5}, output_layer_scale=0.5)
WinningTicket(model, trainer, strategy=strategy).search(rounds=10, epochs=5, prune_fraction=0.2)
```

`scales` multiplies the round's `prune_fraction` per layer type (subclasses included,
first match wins); the output layer uses `output_layer_scale` instead.

The output layer is the last one selected for pruning, which with the default selector is
the last one *defined*. If your model defines its head first, name it:
`LayerwiseMagnitudePruning(output_layer_scale=0.5, output_layer=model.head)`.

By default the weights of every `Linear` and `Conv*d` layer are pruned. Biases and
normalisation layers are left alone. Pass `parameters=` to pick your own.

### Rewinding

```python
WinningTicket(model, trainer)  # rewind to init (the original LTH procedure)
WinningTicket(model, trainer, rewind_step=500)  # late rewinding, needed for deeper conv nets
WinningTicket(model, trainer, rewind="none")  # keep trained weights, restart the LR schedule
WinningTicket(model, trainer, rewind="random", reinit=MyModel)  # IMP with random re-init
```

Late rewinding follows Frankle et al. (2020): after pruning, the weights go back to
those from step *k* and training resumes at step *k*, running the remaining steps with
the learning-rate schedule where it was. `ClassificationTrainer` supports this; a
custom trainer needs a `start_step` argument (see below), and without one each round
replays the whole schedule, with a warning.

**The random re-initialisation control.** Frankle & Carbin's control trains a winning
ticket's masks once from a fresh random initialisation. That is `train_with_masks`:

```python
from lottery import train_with_masks

ticket.search(rounds=6, epochs=5)
control = train_with_masks(LeNet300100(), ticket.masks(), trainer, epochs=5)
```

`ticket.masks()` holds the masks of the last round; record them in a callback's
`on_round_end` to keep every round's (as `examples/mnist_lenet.py --control` does).
`rewind="random"` is a different experiment: an IMP search that re-draws the weights
every round and so finds its own masks. It re-draws each layer with its own
`reset_parameters()`; pass `reinit=` a function that builds a fresh model to draw from
the model's own initialisation instead, which is required when some parameter's module
has no `reset_parameters()` (attention in-projections, positional embeddings, ...).

### Optimisers

`optimiser=` takes anything that builds an optimiser from the parameters, so any torch
optimiser works through `functools.partial`. `sgd()` and `adam()` are shorthands with the
paper's defaults.

```python
from functools import partial

ClassificationTrainer(
    loss_fn, train_loader, test_loader, optimiser=partial(torch.optim.AdamW, lr=3e-4)
)
```

### Your own training loop

`ClassificationTrainer` is one implementation of the `Trainer` protocol: any object with
`fit` and `evaluate` methods like the ones below can drive the search. Use this to train
with [Accelerate](https://huggingface.co/docs/accelerate), Lightning or your own loop. For
example, with Accelerate handling devices, mixed precision and multiple GPUs:

```python
from dataclasses import dataclass
from functools import partial

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from lottery import EpochResult, Metrics
from lottery.training import OptimiserFactory


@dataclass
class AccelerateTrainer:
    """Anything with ``fit`` and ``evaluate`` like this can drive the search."""

    train_loader: DataLoader
    test_loader: DataLoader
    optimiser: OptimiserFactory = partial(torch.optim.AdamW, lr=1e-3)

    def fit(self, model, epochs, on_step=None, on_epoch=None):
        # A fresh optimiser every call: each pruning round restarts the schedule.
        accelerator = Accelerator()  # device placement, mixed precision, multi-GPU
        optimiser = self.optimiser(p for p in model.parameters() if p.requires_grad)
        prepared, optimiser, loader = accelerator.prepare(model, optimiser, self.train_loader)
        results, step = [], 0
        for epoch in range(epochs):
            prepared.train()
            loss_sum, correct, seen = 0.0, 0, 0
            for inputs, targets in loader:
                optimiser.zero_grad()
                outputs = prepared(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)
                accelerator.backward(loss)
                optimiser.step()
                step += 1
                if on_step is not None:  # needed for late rewinding (rewind_step > 0)
                    on_step(step, model)
                loss_sum += loss.item() * len(targets)
                correct += int((outputs.argmax(1) == targets).sum())
                seen += len(targets)
            train = Metrics(loss=loss_sum / seen, accuracy=correct / seen)
            result = EpochResult(epoch, train=train, test=self.evaluate(model, accelerator.device))
            results.append(result)
            if on_epoch is not None:  # optional: lets callbacks see every epoch
                on_epoch(result)
        return results

    @torch.inference_mode()
    def evaluate(self, model, device=None):
        model.eval()
        loss_sum, correct, seen = 0.0, 0, 0
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss_sum += nn.functional.cross_entropy(outputs, targets, reduction="sum").item()
            correct += int((outputs.argmax(1) == targets).sum())
            seen += len(targets)
        return Metrics(loss=loss_sum / seen, accuracy=correct / seen)
```

```python
ticket = WinningTicket(model, AccelerateTrainer(train_loader, test_loader))
```

`fit` must build a fresh optimiser on every call, since each pruning round restarts the
schedule. `on_step` is needed for late rewinding (`rewind_step > 0`), to capture the
weights at step *k*. Two more keyword arguments are optional:

- `on_epoch` lets callbacks such as `ProgressBar` see each epoch.
- `start_step` lets late rewinding resume the schedule at step *k* rather than replaying
  it; see `ClassificationTrainer.fit` for what it should do.

With Lightning, the same shape works: run a new `lightning.Trainer` inside `fit`, and
call `on_step` from a Lightning callback's `on_train_batch_end`.

Under `accelerate launch` every process runs the whole search. Training keeps their
weights in step, but attach file-writing callbacks (`CsvLogger`) and `checkpoint_dir`
only on the main process (`accelerator.is_main_process`), and give every process the
same seed. Multi-GPU runs have not been tested.

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
`torch.load(weights_only=True)`. They also record:

- the rewind settings, which must match on resume
- the pruning strategy (a mismatch warns)
- the CPU and CUDA RNG states, which `load` restores so a resumed run draws the same
  random numbers as an uninterrupted one. The CUDA state is only restored onto the same
  number of GPUs; otherwise `load` warns and leaves it.

A `CsvLogger` pointed at the directory of a resumed run keeps the rows for the rounds
already in the checkpoint and drops any after it.

## Examples

```shell
uv run --group cpu --extra vision python examples/mnist_lenet.py --rounds 10 --epochs 5
uv run --group cpu --extra vision python examples/mnist_lenet.py --rewind random   # compare against the control
uv run --group cu130 --extra vision python examples/cifar10_conv.py --rewind-step 500
uv run --group cpu --extra vision --extra qat python examples/mnist_qat.py
```

Add `--fake-data --limit 120` to any of them for a quick run on random images, without
downloading a dataset; for `cifar10_conv.py` also pass `--rewind-step 1`, since such a
short run never reaches step 500.

A short MNIST run reproduces the paper's qualitative result: rewound tickets hold or
improve accuracy as they get sparser, while the same masks trained from a fresh random
initialisation (Frankle & Carbin's control, `train_with_masks`) degrade. Final accuracy
of each round, as mean ± standard deviation over seeds 0-2:

| density | ticket: val | ticket: test | control: val | control: test |
|--------:|------------:|-------------:|-------------:|--------------:|
| 100%    | 0.943 ± 0.001 | 0.922 ± 0.001 | 0.944 ± 0.003 | 0.922 ± 0.003 |
| 64%     | 0.948 ± 0.001 | 0.926 ± 0.001 | 0.937 ± 0.001 | 0.903 ± 0.004 |
| 41%     | 0.954 ± 0.002 | 0.927 ± 0.003 | 0.934 ± 0.003 | 0.901 ± 0.006 |
| 26%     | 0.955 ± 0.002 | 0.928 ± 0.003 | 0.932 ± 0.004 | 0.898 ± 0.003 |

Choosing by validation accuracy, `best()` picks the sparsest round (26%) in every seed.
To reproduce (10k training images, 2 epochs a round, 6 rounds; `--control` trains each
round's masks again from a fresh initialisation):

```shell
uv run --group cpu --extra vision python examples/mnist_lenet.py --limit 10000 --epochs 2 --rounds 6 --seed 0 --control
```

## Development

The project uses [uv](https://docs.astral.sh/uv/). Choose a PyTorch build with a
dependency group: `cpu` (also right for Apple Silicon) or `cu130` (Linux and Windows).
They are not extras, so installing the package uses whatever torch you already have.

`uv run` re-syncs to the default dependency groups, which would swap the chosen torch build
for PyPI's, so run the tools with `UV_NO_SYNC=1` (as CI does) or `uv run --no-sync`:

```shell
uv sync --group cpu --extra qat --extra vision
export UV_NO_SYNC=1
uv run pytest --cov
uv run ruff check . && uv run ruff format --check .
uv run ty check
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
