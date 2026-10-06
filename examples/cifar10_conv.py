"""Conv-4 on CIFAR-10 with late rewinding (Frankle et al., "Linear Mode Connectivity
and the Lottery Ticket Hypothesis", 2020).

Rewinding to a few hundred steps into training, rather than step 0, is what makes
IMP find tickets for deeper conv nets at standard learning rates.

    uv run --extra cu130 python examples/cifar10_conv.py --rewind-step 500
"""

import argparse
import logging

import torch
from _data import cifar10, device
from torch import nn

from lottery import (
    ClassificationTrainer,
    CsvLogger,
    ProgressBar,
    WinningTicket,
    cosine_annealing,
    sgd,
)
from lottery.models import Conv4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--rewind-step", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/cifar10_conv4")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dev = device()
    train_loader, test_loader = cifar10(limit=args.limit)
    trainer = ClassificationTrainer(
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=train_loader,
        test_loader=test_loader,
        device=dev,
        optimiser=sgd(lr=0.05, momentum=0.9, weight_decay=5e-4),
        scheduler=cosine_annealing,
        autocast_dtype=torch.bfloat16 if dev.type == "cuda" else None,
    )
    ticket = WinningTicket(
        Conv4(),
        trainer,
        rewind_step=args.rewind_step,
        checkpoint_dir=f"{args.output}/checkpoints",
        checkpoint_every=5,
        callbacks=[ProgressBar(), CsvLogger(args.output)],
    )
    result = ticket.search(rounds=args.rounds, epochs=args.epochs, prune_fraction=0.2)
    best = result.best()
    print(f"winning ticket: round {best.round}, {best.density:.2%} of weights")


if __name__ == "__main__":
    main()
