import os
from copy import deepcopy

import torch
import torch.nn as nn

from lottery.WinningTicket import WinningTicket
from lottery.checkpointing.formats import CheckpointFormats
from lottery.pruning import Strategy
from lottery.pruning.LayerPercentile import GlobalPercentilePruningStrategy
from lottery.training import ModelTrainer


class WinningQatTicket(WinningTicket):

    def __init__(self,
                 model: nn.Module,
                 model_trainer: ModelTrainer,
                 device: torch.device,
                 quantized_engine: str = "fbgemm",
                 pruning_strategy: Strategy = GlobalPercentilePruningStrategy(),
                 checkpoint: bool = True,
                 checkpoint_freq: int = 5,
                 base_name: str = "",
                 models_path: str = os.path.join(os.getcwd(), "Results"),
                 enable_logging: bool = True
                 ):
        torch.backends.quantized.engine = quantized_engine
        model_qat = deepcopy(model)
        model_qat.qconfig = torch.quantization.get_default_qat_qconfig(quantized_engine)
        model_prepared = torch.quantization.prepare_qat(model_qat)

        super().__init__(
            model=model_prepared,
            model_trainer=model_trainer,
            device=device,
            pruning_strategy=pruning_strategy,
            checkpoint=checkpoint,
            checkpoint_freq=checkpoint_freq,
            checkpoint_format=CheckpointFormats.Quantised,
            base_name=base_name,
            models_path=models_path,
            enable_logging=enable_logging
        )
