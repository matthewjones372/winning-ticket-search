import unittest

import torch
import torch.nn as nn
from torch.utils.data.dataset import T_co

from lottery.training.ClassificationTrainer import ClassificationTrainer
from lottery.training.OptimiserType import OptimiserType
from lottery.wt.winingticket import WinningTicket
from torch.utils.data import Dataset, DataLoader


class TestDataset(Dataset):
    def __init__(self, shape: tuple, size: int):
        self.shape = shape
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> T_co:
        return torch.randn(self.shape, requires_grad=True), torch.tensor([1])


def test_smoke():
    model = nn.Sequential(
        nn.Linear(5, 1)
    )

    device = torch.device("cpu")

    data = TestDataset((3, 5), 10)

    data_loader = DataLoader(data)

    trainer = ClassificationTrainer(loss_func=nn.NLLLoss(),
                                    optimiser_type=OptimiserType.SGD,
                                    train_loader=data_loader,
                                    test_loader=data_loader,
                                    logging=False,
                                    device=device)

    winning_ticket = WinningTicket(
        model,
        model_trainer=trainer,
        device=device,
        with_logging=False
    )
    initial_weights = winning_ticket.non_zero_weights()

    winning_ticket.search(prune_iterations=1, training_iterations=1, percentage_prune=10)

    final_weights = winning_ticket.non_zero_weights()

    assert final_weights < initial_weights
