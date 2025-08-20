import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange

from src.models.net.AutoEncoder import AutoEncoderModel
from utils.Config import AutoEncoderConfig


class AutoEncoderTrainer:
    def __init__(self, config: AutoEncoderConfig, device: torch.device):
        self.config = config
        self.device = device
        self.architecture = config.architecture
        self.lr = config.lr
        self.lr_decay = config.lr_decay
        self.patience = config.patience
        self.epochs = config.epochs
        self.weights_path = config.weight_path
        self.min_lr = config.min_lr

    def train(self, train_loader: DataLoader, valid_loader: DataLoader | None):
        self.criterion = nn.MSELoss()

        # Get input dimension from first batch
        first_batch = next(iter(train_loader))[0]
        first_batch = first_batch.view(first_batch.size(0), -1)
        input_dim = first_batch.shape[1]

        self.ae_net = AutoEncoderModel(self.architecture, input_dim=input_dim).to(
            self.device
        )

        self.optimizer = optim.Adam(self.ae_net.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        if os.path.exists(self.weights_path):
            self.ae_net.load_state_dict(torch.load(self.weights_path))
            return self.ae_net

        print("Training Autoencoder:")
        t = trange(self.epochs, leave=True)
        for epoch in t:
            train_loss = 0.0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                batch_x = batch_x.view(batch_x.size(0), -1)
                self.optimizer.zero_grad()
                output = self.ae_net(batch_x)
                loss = self.criterion(output, batch_x)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                t.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                    }
                )

            train_loss /= len(train_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            if current_lr <= self.min_lr:
                break

            t.set_description(
                "Train Loss: {:.7f}, LR: {:.6f}".format(train_loss, current_lr)
            )
            t.refresh()

        torch.save(self.ae_net.state_dict(), self.weights_path)

    def validate(self, valid_loader: DataLoader) -> float:
        self.ae_net.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_x in valid_loader:
                batch_x = batch_x.to(self.device)
                batch_x = batch_x.view(batch_x.size(0), -1)
                output = self.ae_net(batch_x)
                loss = self.criterion(output, batch_x)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        return valid_loss

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        print("Embedding data ...")
        self.ae_net.eval()
        with torch.no_grad():
            X = X.view(X.size(0), -1)
            encoded_data = self.ae_net.encode(X.to(self.device))
        return encoded_data
