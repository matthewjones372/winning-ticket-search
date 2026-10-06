"""LeNet-300-100 on MNIST, the headline experiment from Frankle & Carbin (2019).

uv run --group cpu --extra vision python examples/mnist_lenet.py --rounds 10 --epochs 5
"""

import argparse
import csv
import logging
from pathlib import Path

import torch
from _data import device, mnist
from torch import nn

from lottery import (
    Callback,
    ClassificationTrainer,
    CsvLogger,
    LayerwiseMagnitudePruning,
    ProgressBar,
    RoundResult,
    WinningTicket,
    adam,
    train_with_masks,
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
    parser.add_argument(
        "--control",
        action="store_true",
        help="also train each round's masks from a fresh init (Frankle & Carbin's control)",
    )
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
        callbacks=[ProgressBar(), CsvLogger(args.output), masks := MaskRecorder()],
    )
    result = ticket.search(
        rounds=args.rounds, epochs=args.epochs, prune_fraction=args.prune_fraction
    )

    best = result.best()
    print(f"winning ticket: round {best.round}, {best.density:.2%} of weights")

    if args.control:
        # The same masks, trained once from a new random initialisation.
        with (Path(args.output) / "control.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["round", "density", "val_accuracy", "test_accuracy"])
            for r in result.rounds:
                final = train_with_masks(
                    LeNet300100(), masks.by_round[r.round], trainer, args.epochs
                )[-1]
                val = final.val.accuracy if final.val is not None else float("nan")
                writer.writerow([r.round, r.density, val, final.test.accuracy])
                print(
                    f"control round {r.round}  density {r.density:.2%}  "
                    f"val acc {val:.4f}  test acc {final.test.accuracy:.4f}"
                )


class MaskRecorder(Callback):
    """Keeps each round's masks, to train them again from scratch afterwards."""

    def __init__(self) -> None:
        self.by_round: dict[int, dict[str, torch.Tensor]] = {}

    def on_round_end(self, ticket: WinningTicket, result: RoundResult) -> None:
        self.by_round[result.round] = ticket.masks()


if __name__ == "__main__":
    main()
