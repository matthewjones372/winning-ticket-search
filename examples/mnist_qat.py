"""Lottery ticket search with quantisation-aware training via torchao.

Each round trains the fake-quantised network and also reports the accuracy of the
truly int8-quantised model.

    uv run --group cpu --extra vision --extra qat python examples/mnist_qat.py --rounds 5 --epochs 3
"""

import argparse
import logging

import torch
from _data import mnist
from torch import nn

from lottery import ClassificationTrainer, CsvLogger, ProgressBar, adam
from lottery.models import LeNet300100
from lottery.qat import QatWinningTicket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/mnist_qat")
    parser.add_argument(
        "--fake-data", action="store_true", help="random images instead of downloading"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)  # weights, shuffling and random re-init
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    train_loader, val_loader, test_loader = mnist(limit=args.limit, fake=args.fake_data)
    # The default int8 config runs on CPU; pass base_config= for GPU int4/fp8 schemes.
    trainer = ClassificationTrainer(
        nn.CrossEntropyLoss(),
        train_loader,
        test_loader,
        optimiser=adam(lr=1.2e-3),
        val_loader=val_loader,
    )
    ticket = QatWinningTicket(
        LeNet300100(), trainer, callbacks=[ProgressBar(), CsvLogger(args.output)]
    )
    # Each round's log line carries the fake-quantised and the real int8 accuracy.
    ticket.search(rounds=args.rounds, epochs=args.epochs)


if __name__ == "__main__":
    main()
