import torch.nn as nn

from lottery.checkpointing.checkpointformat import CheckpointFormat


class StateScript(CheckpointFormat):
    @staticmethod
    def save(model: nn.Module, model_path: str) -> None:
        print(f"Saving state to {model_path}")
        torch.jit.save(torch.jit.script(model), model_path)
