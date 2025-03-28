import copy
from typing import Optional

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange

from data.LoadData import load_dataset
from src.models.net.Embedding import ManifoldEmbedingNetwork
from utils.Config import Config
from utils.Loss import TotalCodingRate
from utils.Utils import save_ckpt, save_state


class ManifoldsEmbeddingTrainer:
    def __init__(self, config: Config, device: torch.device):
        self.device = device
        self.config = config
        self.dataset = self.config.dataset.dataset
        self.lr = self.config.training.lr
        self.backbone = self.config.model.backbone
        self.feature_dims = self.config.model.feature_dims
        self.z_dims = self.config.model.z_dims
        self.batch_size = self.config.dataset.batch_size
        self.eps = self.config.optimization.eps
        self.momentum = self.config.optimization.momentum
        self.weight_decay = self.config.optimization.weight_decay
        self.n_epoch = self.config.training.num_epoch
        self.weight_path = self.config.training.weight_path
        self.save_every = self.config.training.save_every

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
            trainset, valset = load_dataset(self.dataset)
            is_feature = False
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, is_feature

    def train(self, X: Optional[torch.Tensor], y: Optional[torch.Tensor]):
        self.criterion = TotalCodingRate(eps=self.eps)
        if X is not None:
            self.feature_dims = X.shape[1]
        self.net = ManifoldEmbedingNetwork(
            backbone_name=self.backbone,
            feature_dims=self.feature_dims,
            z_dims=self.z_dims,
        )
        self.net.to(self.device)
        para_list = [p for p in self.net.subspace.parameters()]

        optimizer = optim.SGD(
            para_list,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=False,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epoch, eta_min=0, last_epoch=-1
        )

        train_loader, val_loader, is_feature = self._get_data_loader(X, y)

        print("Training Embedding: ")
        t = trange(self.n_epoch, leave=True)
        for epoch in t:
            for step, (x, y) in enumerate(train_loader):
                x, y = x.float().to(self.device), y.to(self.device)
                z = self.net(x, detach_feature=True, is_feature=is_feature)

                z_list = z.chunk(2, dim=0)
                loss = (self.criterion(z_list[0]) + self.criterion(z_list[1])) / 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                save_state(self.weight_path, epoch, step, loss.item())

                t.set_description(str(epoch))
                t.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

            scheduler.step()
            if (epoch + 1) % self.save_every == 0:
                save_ckpt(self.weight_path, self.net, optimizer, scheduler, epoch + 1)

        print("training complete.")
