import copy
from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from data.LoadData import load_dataset
from utils.Config import Config
from utils.logger import Logger


class PaddedDataset(Dataset):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.original_length = len(dataset)
        # Calculate how many samples we need to add to make the last batch full
        self.padded_length = (
            (self.original_length + self.batch_size - 1) // self.batch_size
        ) * self.batch_size

    def __len__(self):
        return self.padded_length

    def __getitem__(self, idx):
        if idx < self.original_length:
            return self.dataset[idx]
        else:
            # For indices beyond the original length, wrap around to the beginning
            return self.dataset[idx % self.original_length]


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

        padded_train_dataset = PaddedDataset(trainset, self.batch_size)
        padded_valid_dataset = PaddedDataset(valset, self.batch_size)

        train_loader = DataLoader(
            padded_train_dataset, batch_size=self.batch_size, shuffle=True
        )
        ortho_loader = DataLoader(
            padded_train_dataset, batch_size=self.batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            padded_valid_dataset, batch_size=self.batch_size, shuffle=False
        )
        return train_loader, ortho_loader, valid_loader, is_feature
        # train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        # val_loader = DataLoader(valset, batch_size=self.batch_size, shuffle=False)
        # return train_loader, val_loader, is_feature

    @abstractmethod
    def train(self):
        raise NotImplementedError("Trainer must implement the train method")
        raise NotImplementedError("Trainer must implement the train method")
