import numpy as np

import torch
from torch import nn as nn

from lottery.pruning.Strategy import Strategy


class GlobalPercentilePruningStrategy(Strategy):

    def prune(self, model: nn.Module,
              mask: [np.array],
              percentage_prune: int,
              prune_epoch: int,
              device: torch.device) -> ([np.array], nn.Module):
        i = 0
        for name, layer in model.named_parameters():
            if 'weight' in name:
                tensor = layer.data.cpu().numpy()
                non_zero = tensor[np.nonzero(tensor)]
                percentile = np.percentile(abs(non_zero), percentage_prune)

                filtered_mask = np.where(abs(tensor) < percentile, 0, mask[i])
                layer.data = torch.from_numpy(tensor * filtered_mask).to(device)
                mask[i] = filtered_mask
                i += 1
        return mask, model
