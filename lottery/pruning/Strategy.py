import abc
import torch
import torch.nn as nn
import numpy as np


class Strategy(abc.ABC):
    @abc.abstractmethod
    def prune(self,
              model: nn.Module,
              mask: [np.array],
              percentage_prune: int,
              prune_epoch: int,
              device: torch.device) -> ([np.array], nn.Module):
        pass
