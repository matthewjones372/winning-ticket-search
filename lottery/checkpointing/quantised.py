import torch
import torch.nn as nn
from copy import deepcopy

from lottery.checkpointing.checkpointformat import CheckpointFormat


class Quantised(CheckpointFormat):
    @staticmethod
    def save(model: nn.Module, model_path: str):
        print(f"Saving state to {model_path}")
        model_int8 = deepcopy(model)
        model_int8 = torch.quantization.convert(model_int8.to(torch.device("cpu")))
        torch.jit.save(torch.jit.script(model_int8), model_path)
