"""Lottery ticket search with quantisation-aware training via torchao.

Each round trains the fake-quantised network and also reports the accuracy of the
truly int8-quantised model.

    uv run --extra cpu --extra qat python examples/mnist_qat.py --rounds 5 --epochs 3
"""

import argparse
import logging

from _data import mnist
from torch import nn

from lottery import ClassificationTrainer, adam
from lottery.models import LeNet300100
from lottery.qat import QatWinningTicket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/mnist_qat")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    train_loader, test_loader = mnist(limit=args.limit)
    # The default int8 config runs on CPU; pass base_config= for GPU int4/fp8 schemes.
    trainer = ClassificationTrainer(
        nn.CrossEntropyLoss(), train_loader, test_loader, optimiser=adam(lr=1.2e-3)
    )
    ticket = QatWinningTicket(LeNet300100(), trainer, output_dir=args.output)
    result = ticket.search(rounds=args.rounds, epochs=args.epochs)

    for r in result.rounds:
        q = r.extra_metrics["quantised"]
        print(
            f"round {r.round:2d}  density {r.density:6.2%}  "
            f"fake-quant acc {r.final_test.accuracy:.4f}  int8 acc {q.accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
