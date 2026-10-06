"""LeNet-300-100 on MNIST, the headline experiment from Frankle & Carbin (2019).

uv run --extra cpu python examples/mnist_lenet.py --rounds 10 --epochs 5
"""

import argparse
import logging

from _data import device, mnist
from torch import nn

from lottery import ClassificationTrainer, LayerwiseMagnitudePruning, WinningTicket, adam
from lottery.models import LeNet300100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--prune-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None, help="subsample for a quick run")
    parser.add_argument("--rewind", default="weights", choices=["weights", "random", "none"])
    parser.add_argument("--output", default="results/mnist_lenet")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    train_loader, test_loader = mnist(limit=args.limit)
    trainer = ClassificationTrainer(
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=train_loader,
        test_loader=test_loader,
        device=device(),
        optimiser=adam(lr=1.2e-3),  # the paper's setting for LeNet
    )
    ticket = WinningTicket(
        LeNet300100(),
        trainer,
        # The paper prunes the output layer at half the rate of the hidden layers.
        strategy=LayerwiseMagnitudePruning(output_layer_scale=0.5),
        rewind=args.rewind,
        output_dir=args.output,
    )
    result = ticket.search(
        rounds=args.rounds, epochs=args.epochs, prune_fraction=args.prune_fraction
    )

    for r in result.rounds:
        print(f"round {r.round:2d}  density {r.density:6.2%}  test acc {r.final_test.accuracy:.4f}")
    best = result.best()
    print(f"winning ticket: round {best.round}, {best.density:.2%} of weights")


if __name__ == "__main__":
    main()
