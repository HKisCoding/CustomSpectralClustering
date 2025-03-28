import copy
from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from data.LoadData import load_dataset
from utils.Config import Config
from utils.logger import Logger


class BaseTrainer(object):
    def __init__(self, config: Union[Dict[str, Any], Config]):
        if isinstance(config, dict):
            self.config = Config.from_dict(config)
        else:
            self.config = config
        self.logger = Logger(name=self.__class__.__name__)
        self.dataset_name = self.config.dataset.dataset
        self.batch_size = self.config.dataset.batch_size

    def _get_data_loader(self, X: Optional[torch.Tensor], y: Optional[torch.Tensor]):
        if X is not None and y is not None:
            train_size = int(0.8 * len(X))
            valid_size = len(X) - train_size
            dataset = TensorDataset(X, y)
            train_dataset, valid_dataset = random_split(
                dataset, [train_size, valid_size]
            )
            trainset, valset = (
                copy.deepcopy(train_dataset),
                copy.deepcopy(valid_dataset),
            )
            is_feature = True
        else:
            trainset, valset = load_dataset(self.dataset_name)
            is_feature = False
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, is_feature

    @abstractmethod
    def train(self):
        raise NotImplementedError("Trainer must implement the train method")
