"""Shared dataset helpers for the examples (needs the `vision` extra)."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

type Loaders = tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]
"""Train, validation and test."""


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(data: Dataset[Any], batch_size: int, shuffle: bool) -> DataLoader[Any]:
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )


def _loaders(
    train: Dataset[Any],
    held_out: Dataset[Any],
    test: Dataset[Any],
    batch_size: int,
    limit: int | None,
) -> Loaders:
    """Train, validation and test loaders.

    The last twelfth of the training set (5,000 MNIST images, as in Frankle & Carbin) is
    held out for validation. ``held_out`` is the training set again, but with the test
    transforms, so validation images are not augmented.
    """
    size = len(train)  # type: ignore[arg-type]
    n_val = size // 12
    train_idx, val_idx, test_idx = range(size - n_val), range(size - n_val, size), None
    if limit is not None:
        train_idx, val_idx, test_idx = train_idx[:limit], val_idx[: limit // 5], range(limit // 5)
    return (
        _loader(Subset(train, train_idx), batch_size, shuffle=True),
        _loader(Subset(held_out, val_idx), 1024, shuffle=False),
        _loader(test if test_idx is None else Subset(test, test_idx), 1024, shuffle=False),
    )


def _fake(shape: tuple[int, int, int], size: int) -> tuple[Dataset[Any], Dataset[Any]]:
    """Random images in the real dataset's shape, so the examples run without downloads."""
    tf = transforms.ToTensor()
    return (
        datasets.FakeData(size, shape, num_classes=10, transform=tf, random_offset=0),
        datasets.FakeData(size // 5, shape, num_classes=10, transform=tf, random_offset=size),
    )


def mnist(batch_size: int = 60, limit: int | None = None, fake: bool = False) -> Loaders:
    if fake:
        train, test = _fake((1, 28, 28), limit or 1200)
        return _loaders(train, train, test, batch_size, None)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    return _loaders(train, train, test, batch_size, limit)


def cifar10(batch_size: int = 60, limit: int | None = None, fake: bool = False) -> Loaders:
    if fake:
        train, test = _fake((3, 32, 32), limit or 1200)
        return _loaders(train, train, test, batch_size, None)
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    held_out = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=test_tf)
    test = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    return _loaders(train, held_out, test, batch_size, limit)
