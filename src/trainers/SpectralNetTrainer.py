import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange

from src.models.net.SpectralNet import SpectralNetModel
from utils.Config import Config
from utils.Loss import SpectralClusterLoss
from utils.Utils import (
    compute_scale,
    get_gaussian_kernel,
    get_nearest_neighbors,
    make_batch_for_sparse_grapsh,
)


class SpectralNetTrainer:
    def __init__(self, config: Config, device: torch.device, is_sparse: bool):
        self.device = device
        self.is_sparse = is_sparse
        self.config = config
        self.lr = self.config.training.lr
        self.n_nbg = self.config.spectral.n_nbg
        self.min_lr = self.config.spectral.min_lr
        self.epochs = self.config.training.num_epoch
        self.scale_k = self.config.spectral.scale_k
        self.lr_decay = self.config.spectral.lr_decay
        self.patience = self.config.spectral.patience
        self.architecture = self.config.spectral.architecture
        self.batch_size = self.config.dataset.batch_size
        self.is_local_scale = self.config.spectral.is_local_scale

    def create_affingity_matrix_from_scale(
        self, X: torch.Tensor, scale: float
    ) -> torch.Tensor:
        is_local = self.is_local_scale
        n_neighbors = self.n_nbg
        Dx = torch.cdist(X, X)
        Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
        # scale = compute_scale(Dis, k=scale, is_local=is_local)
        W = get_gaussian_kernel(
            Dx, scale, indices, device=self.device, is_local=is_local
        )
        return W

    def _get_affinity_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        This function computes the affinity matrix W using the Gaussian kernel.

        Args:
            X (torch.Tensor):   The input data

        Returns:
            torch.Tensor: The affinity matrix W
        """

        is_local = self.is_local_scale
        n_neighbors = self.n_nbg
        scale_k = self.scale_k
        Dx = torch.cdist(X, X)
        Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
        scale = compute_scale(Dis, k=scale_k, is_local=is_local)
        W = get_gaussian_kernel(
            Dx, scale, indices, device=self.device, is_local=is_local
        )
        return W

    def _get_data_loader(self) -> tuple:
        """
        This function returns the data loaders for training, validation and testing.

        Returns:
            tuple:  The data loaders
        """
        if self.y is None:
            self.y = torch.zeros(len(self.X))
        train_size = int(0.9 * len(self.X))
        valid_size = len(self.X) - train_size
        dataset = TensorDataset(self.X, self.y)
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        ortho_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False
        )
        return train_loader, ortho_loader, valid_loader

    def train(self, X: torch.Tensor, y: torch.Tensor):
        # Flatten the input tensor
        self.X = X.view(X.size(0), -1)
        self.y = y

        self.counter = 0
        self.criterion = SpectralClusterLoss()

        self.spectral_net = SpectralNetModel(
            self.architecture, input_dim=self.X.shape[1]
        ).to(self.device)

        self.optimizer = optim.Adam(self.spectral_net.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        train_loader, ortho_loader, valid_loader = self._get_data_loader()

        print("Training SpectralNet:")
        t = trange(self.epochs, leave=True)
        total_train_loss = []
        total_val_loss = []
        for epoch in t:
            train_loss = 0.0
            for (X_grad, _), (X_orth, _) in zip(train_loader, ortho_loader):
                X_grad = X_grad.to(device=self.device)
                X_grad = X_grad.view(X_grad.size(0), -1)
                X_orth = X_orth.to(device=self.device)
                X_orth = X_orth.view(X_orth.size(0), -1)

                if self.is_sparse:
                    X_grad = make_batch_for_sparse_grapsh(X_grad)
                    X_orth = make_batch_for_sparse_grapsh(X_orth)

                # Orthogonalization step
                self.spectral_net.eval()
                self.spectral_net(X_orth, True)

                # Gradient step
                self.spectral_net.train()
                self.optimizer.zero_grad()

                Y = self.spectral_net(X_grad, False)

                W = self._get_affinity_matrix(X_grad)

                loss = self.criterion(W, Y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation step
            valid_loss = self.validate(valid_loader)
            self.scheduler.step(valid_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.config.spectral.min_lr:
                break
            t.set_description(
                "Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(
                    train_loss, valid_loss, current_lr
                )
            )
            total_train_loss.append(train_loss)
            total_val_loss.append(valid_loss)
            t.refresh()
        train_result = {"train_loss": total_train_loss, "val_loss": total_val_loss}

        return self.spectral_net, train_result
