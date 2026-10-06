import pytest
import torch

from lottery import WinningTicket
from lottery.models import Conv2, Conv4, Conv6, LeNet300100

from .conftest import ShiftTrainer


def test_lenet_shape():
    assert LeNet300100()(torch.randn(2, 1, 28, 28)).shape == (2, 10)


@pytest.mark.parametrize(("cls", "convs"), [(Conv2, 2), (Conv4, 4), (Conv6, 6)])
def test_conv_nets(cls, convs):
    model = cls(num_classes=7)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 7)
    ticket = WinningTicket(model, ShiftTrainer(), progress=False)
    assert len(ticket.parameters) == convs + 3
