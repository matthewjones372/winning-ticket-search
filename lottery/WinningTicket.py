import datetime
import os
from copy import deepcopy
from dataclasses import asdict
from math import ceil
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import tqdm

from lottery.checkpointing.formats import CheckpointFormats
from lottery.checkpointing.resolver import CheckpointFormatResolver
from lottery.metrics.DataLogger import DataLogger
from lottery.metrics.ResultTypes import EpochResult
from lottery.pruning.LayerPercentile import GlobalPercentilePruningStrategy
from lottery.pruning.Strategy import Strategy
from lottery.training.ModelTrainer import ModelTrainer

WEIGHT_LOGGER_HEADERS = ["epoch", "layer_name", "non_zero_weights"]
METRICS_LOGGER_HEADERS = ["epoch", "accuracy", "loss"]


class WinningTicket:
    def __init__(
            self,
            model: nn.Module,
            model_trainer: ModelTrainer,
            device: torch.device,
            pruning_strategy: Strategy = GlobalPercentilePruningStrategy(),
            checkpoint: bool = True,
            checkpoint_freq: int = 5,
            checkpoint_format: CheckpointFormats = CheckpointFormats.Script,
            base_name: str = "",
            models_path: str = os.path.join(os.getcwd(), "Results"),
            enable_logging: bool = True
    ):
        self.model = model.to(device)
        self.model_trainer = model_trainer
        self.initial_state_dict = deepcopy(model.state_dict())
        self.initial_mask = self.__create_initial_mask(model)
        self.mask = self.initial_mask
        self.device = device
        self.checkpoint = checkpoint
        self.checkpoint_save_format = CheckpointFormatResolver(checkpoint_format).resolve()
        self.checkpoint_frequency = checkpoint_freq
        self.base_name = base_name
        self.base_path = os.path.join(models_path, base_name)
        self.pruning_strategy = pruning_strategy
        self.enable_logging = enable_logging

        if not os.path.exists(self.base_path) and (enable_logging or checkpoint):
            os.makedirs(self.base_path)

        current_date = str(datetime.datetime.now().strftime("%Y_%m_%d"))
        weights_file_path = f"{base_name}_weights_{current_date}.csv"

        if enable_logging:
            print(f"Writing weight metrics to {weights_file_path}")
            self.weight_logger = DataLogger(
                WEIGHT_LOGGER_HEADERS,
                weights_file_path,
                self.base_path,
            )

            metrics_file_path = f"{base_name}_metrics_{current_date}.csv"
            print(f"Writing test metrics to {metrics_file_path}")
            self.metrics_logger = DataLogger(
                METRICS_LOGGER_HEADERS, metrics_file_path, self.base_path
            )

    def non_zero_weights(self):
        self.model.eval()
        non_zero = 0

        for name, layer in self.model.named_parameters():
            if self.__is_weight(name):
                tensor = layer.data.cpu().numpy()
                non_zero_count = np.count_nonzero(tensor)
                non_zero += non_zero_count

        return non_zero

    def update_non_zero_weights(
            self, epoch: int, _model: nn.Module, logging: bool = True
    ):

        _model.eval()
        weight_layer: List[(int, str, float)] = []
        nonzero = 0
        total = 0
        for name, layer in _model.named_parameters():
            if self.__is_weight(name):
                tensor = layer.data.cpu().numpy()
                non_zero_count = np.count_nonzero(tensor)
                total_params = np.prod(tensor.shape)
                nonzero += non_zero_count
                total += total_params
                weight_layer.append((epoch, name, nonzero))

                if logging:
                    print(
                        f"{name:20} | non zeros={non_zero_count:4,}/{total_params:4,}\n"
                        f"total_pruned = {total_params - non_zero_count :4,}\n"
                        f"alive: {nonzero:,} | pruned : {total - nonzero:,}\n"
                        f"total: {total:,} | ratio: {nonzero / total:.3f}\n"
                    )

        return weight_layer

    def search_by_target_sparsity(
            self,
            training_iterations: int = 30,
            target_sparsity: int = 5,
            percentage_prune: int = 5,
    ):
        prune_iterations = ceil(
            -np.log(target_sparsity / 100.0) / percentage_prune * 100
        )
        return self.search(prune_iterations, training_iterations, percentage_prune)

    def search(
            self,
            prune_iterations: int = 30,
            training_iterations: int = 30,
            percentage_prune: int = 10,
    ):

        model = self.model
        mask = self.mask
        loop_message = "###### pruning Iteration #######"
        for prune_iteration in tqdm.tqdm(
                range(prune_iterations + 1), desc=loop_message, position=0, leave=True
        ):
            if prune_iteration == 0:
                mask, model = self.__set_models_to_original_init(model, mask)
            else:
                mask, model = self.pruning_strategy.prune(
                    model, mask, percentage_prune, prune_iteration, device=self.device
                )

                i = 0
                model = model.apply(self.__initialise_weight)

                for name, layer in model.named_parameters():
                    if self.__is_weight(name):
                        update_layer_data = layer.data.cpu().numpy() * mask[i]
                        layer.data = torch.from_numpy(update_layer_data).to(
                            layer.device
                        )
                        i += 1

            result = self.model_trainer.train_and_test(model, training_iterations)
            epoch_results, model = result.results, result.model

            if self.enable_logging:
                self.__update_metrics_log(prune_iteration, epoch_results)
                weight_layer = self.update_non_zero_weights(prune_iteration, model)
                self.weight_logger.write_rows(weight_layer)

            if self.checkpoint and prune_iteration % self.checkpoint_frequency == 0:
                self.save_state(model, prune_iteration)

        self.model = model
        self.mask = mask

        return self

    def __update_metrics_log(
            self, epoch: int, epoch_results: List[EpochResult]
    ) -> None:
        def as_row(result: EpochResult):
            metrics = asdict(result.test_result)
            metrics["epoch"] = epoch
            return metrics

        test_results = list(map(lambda r: as_row(r), epoch_results))
        self.metrics_logger.write_rows(test_results, map_headers=False)

    def __set_models_to_original_init(self, _model: nn.Module, _mask):
        i = 0
        for name, param in _model.named_parameters():
            if self.__is_weight(name):
                it = _mask[i] * self.initial_state_dict[name].cpu().numpy()
                param.data = torch.from_numpy(it).to(self.device)
                i += 1

            if self.__is_bias(name):
                data = self.initial_state_dict[name]
                param.data = data.to(self.device)

        return _mask, _model

    def __create_initial_mask(self, _model: nn.Module):
        mask = [None for name, _ in _model.named_parameters() if self.__is_weight(name)]

        weights = 0
        for name, param in _model.named_parameters():
            if self.__is_weight(name):
                tensor = param.data.cpu().numpy()
                mask[weights] = np.ones_like(tensor)
                weights += 1

        return mask

    @staticmethod
    def __initialise_weight(layer: nn.Module):
        if isinstance(layer, nn.Conv1d):
            init.normal_(layer.weight.data)
            if layer.bias is not None:
                init.normal_(layer.bias.data)

        if isinstance(layer, nn.Conv2d):
            init.xavier_normal_(layer.weight.data)
            if layer.bias is not None:
                init.normal_(layer.bias.data)

        if isinstance(layer, nn.BatchNorm1d):
            init.normal_(layer.weight.data, mean=1, std=0.02)
            init.constant_(layer.bias.data, 0)

        if isinstance(layer, nn.BatchNorm2d):
            init.normal_(layer.weight.data, mean=1, std=0.02)
            init.constant_(layer.bias.data, 0)

        if isinstance(layer, nn.BatchNorm3d):
            init.normal_(layer.weight.data, mean=1, std=0.02)
            init.constant_(layer.bias.data, 0)

        if isinstance(layer, nn.Linear):
            init.xavier_normal_(layer.weight.data)
            init.normal_(layer.bias.data)

        if isinstance(layer, nn.LSTM):
            for param in layer.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

        if isinstance(layer, nn.LSTMCell):
            for param in layer.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

        if isinstance(layer, nn.GRU):
            for param in layer.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

        if isinstance(layer, nn.GRUCell):
            for param in layer.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

    @staticmethod
    def __is_weight(parameter_name: str):
        return "weight" in parameter_name

    @staticmethod
    def __is_bias(parameter_name: str):
        return "bias" in parameter_name

    def save_state(
            self, model: nn.Module, checkpoint_num: Optional[int] = None
    ) -> None:
        state_save_dir = os.path.join(self.base_path, self.base_name)

        if not os.path.isdir(state_save_dir):
            os.makedirs(state_save_dir)

        model_name = f"{self.base_name}_{checkpoint_num}.pt"
        model_path = os.path.join(state_save_dir, model_name)
        self.checkpoint_save_format.save(model, model_path)

    def __call__(self, data):
        return self.model(data)

    def __repr__(self):
        return f"{self.model} Winning-Ticket Search"
