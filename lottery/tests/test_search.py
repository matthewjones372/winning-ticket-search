import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lottery.WinningTicket import WinningTicket
from lottery.checkpointing.quantised import Quantised
from lottery.training.ClassificationTrainer import ClassificationTrainer
from lottery.training.OptimiserType import OptimiserType
from lottery.utils.testing.dataset import TestDataset


def test_smoke():
    model = nn.Sequential(nn.Linear(5, 1))

    device = torch.device("cpu")

    data = TestDataset((3, 5), 10)

    data_loader = DataLoader(data)

    trainer = ClassificationTrainer(
        loss_func=nn.NLLLoss(),
        optimiser_type=OptimiserType.SGD,
        train_loader=data_loader,
        test_loader=data_loader,
        enable_logging=False,
        device=device,
    )

    winning_ticket = WinningTicket(
        model,
        model_trainer=trainer,
        device=device,
        checkpoint=False,
        enable_logging=False
    )
    initial_weights = winning_ticket.non_zero_weights()

    winning_ticket.search(
        prune_iterations=1, training_iterations=1, percentage_prune=10
    )

    final_weights = winning_ticket.non_zero_weights()

    assert final_weights < initial_weights


def test_can_save_and_load_qat_model():
    qat_model = nn.Sequential(torch.quantization.QuantStub(), nn.Linear(5, 1), torch.quantization.DeQuantStub())
    file_path = os.path.join(os.getcwd(), "temp.pt")
    Quantised.save(qat_model, file_path)
    loaded_model = Quantised.load(file_path, torch.device("cpu"))
    os.remove(file_path)
    assert loaded_model
