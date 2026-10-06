"""Shared dataset helpers for the examples (needs the `vision` extra)."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loaders(train, test, batch_size: int, limit: int | None) -> tuple[DataLoader, DataLoader]:
    if limit is not None:
        train, test = Subset(train, range(limit)), Subset(test, range(limit // 5))
    pin = torch.cuda.is_available()
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin),
        DataLoader(test, batch_size=1024, shuffle=False, num_workers=2, pin_memory=pin),
    )


def mnist(batch_size: int = 60, limit: int | None = None) -> tuple[DataLoader, DataLoader]:
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    return _loaders(train, test, batch_size, limit)


def cifar10(batch_size: int = 60, limit: int | None = None) -> tuple[DataLoader, DataLoader]:
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
    test = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    return _loaders(train, test, batch_size, limit)
