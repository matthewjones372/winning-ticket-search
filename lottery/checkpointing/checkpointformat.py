import abc

import torch
import torch.nn as nn


class CheckpointFormat(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def save(model: nn.Module, model_path: str) -> None:
        pass

    @staticmethod
    def load(file_path: str, device: torch.device) -> nn.Module:
        load_model = torch.jit.load(file_path, map_location=device)
        return load_model
