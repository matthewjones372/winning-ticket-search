"""LeNet-300-100 on MNIST, the headline experiment from Frankle & Carbin (2019).

uv run --group cpu --extra vision python examples/mnist_lenet.py --rounds 10 --epochs 5
"""

import argparse
import logging

import torch
from _data import device, mnist
from torch import nn

from lottery import (
    ClassificationTrainer,
    CsvLogger,
    LayerwiseMagnitudePruning,
    ProgressBar,
    WinningTicket,
    adam,
)
from lottery.models import LeNet300100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--prune-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None, help="subsample for a quick run")
    parser.add_argument("--rewind", default="weights", choices=["weights", "random", "none"])
    parser.add_argument("--output", default="results/mnist_lenet")
    parser.add_argument(
        "--fake-data", action="store_true", help="random images instead of downloading"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)  # weights, shuffling and random re-init
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    train_loader, val_loader, test_loader = mnist(limit=args.limit, fake=args.fake_data)
    trainer = ClassificationTrainer(
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        device=device(),
        optimiser=adam(lr=1.2e-3),  # the paper's setting for LeNet
    )
    ticket = WinningTicket(
        LeNet300100(),
        trainer,
        # The paper prunes the output layer at half the rate of the hidden layers.
        strategy=LayerwiseMagnitudePruning(output_layer_scale=0.5),
        rewind=args.rewind,
        callbacks=[ProgressBar(), CsvLogger(args.output)],
    )
    result = ticket.search(
        rounds=args.rounds, epochs=args.epochs, prune_fraction=args.prune_fraction
    )

    best = result.best()
    print(f"winning ticket: round {best.round}, {best.density:.2%} of weights")


if __name__ == "__main__":
    main()
