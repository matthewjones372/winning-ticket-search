import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class TestDataset(Dataset):
    def __init__(self, shape: tuple, size: int):
        self.shape = shape
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> T_co:
        return torch.randn(self.shape, requires_grad=True), torch.tensor([1])
