import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import trange

from src.models.net.SiameseNet import SiameseNetModel
from src.models.net.SpectralNet import SpectralNetModel
from src.trainers.BaseTrainer import BaseTrainer
from src.trainers.SiameseNetTrainer import SiameseTrainer
from utils.Config import Config
from utils.Loss import SpectralNetLoss
from utils.Utils import (
    compute_scale,
    get_clusters_by_kmeans,
    get_gaussian_kernel,
    get_nearest_neighbors,
    make_batch_for_sparse_grapsh,
)


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


class SpectralNetTrainer(BaseTrainer):
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

    # def _get_data_loader(self) -> tuple:
    #     """
    #     This function returns the data loaders for training, validation and testing.
    #     The last batch will be padded by repeating samples from the beginning of the dataset.

    #     Returns:
    #         tuple:  The data loaders
    #     """
    #     if self.y is None:
    #         self.y = torch.zeros(len(self.X))
    #     train_size = int(0.9 * len(self.X))
    #     valid_size = len(self.X) - train_size
    #     dataset = TensorDataset(self.X, self.y)
    #     train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    #     # Create padded versions of the datasets
    #     padded_train_dataset = PaddedDataset(train_dataset, self.batch_size)
    #     padded_valid_dataset = PaddedDataset(valid_dataset, self.batch_size)

    #     train_loader = DataLoader(
    #         padded_train_dataset, batch_size=self.batch_size, shuffle=True
    #     )
    #     ortho_loader = DataLoader(
    #         padded_train_dataset, batch_size=self.batch_size, shuffle=True
    #     )
    #     valid_loader = DataLoader(
    #         padded_valid_dataset, batch_size=self.batch_size, shuffle=False
    #     )
    #     return train_loader, ortho_loader, valid_loader

    def train_with_siamese_net(
        self, X: torch.Tensor, y: torch.Tensor, siamese_weights: None | str = None
    ):
        # Flatten the input tensor
        self.X = X.view(X.size(0), -1)
        self.y = y

        # Create and train SiameseNet if weights are None, otherwise load weights
        siamese_trainer = SiameseTrainer(self.config.siamese, self.device)
        if siamese_weights is None:
            siamese_net = siamese_trainer.train(self.X)
        else:
            siamese_net = SiameseNetModel(
                self.config.siamese.architecture, input_dim=self.X.shape[1]
            ).to(self.device)
            siamese_net.load_state_dict(torch.load(siamese_weights))

        self.counter = 0
        self.criterion = SpectralNetLoss()

        self.spectral_net = SpectralNetModel(
            self.architecture, input_dim=self.X.shape[1]
        ).to(self.device)

        self.optimizer = optim.Adam(self.spectral_net.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        train_loader, ortho_loader, valid_loader, _ = self._get_data_loader(
            X=self.X, y=self.y
        )

        print("Training Spectral Network with SiameseNet:")
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
                self.spectral_net(X_orth, should_update_orth_weights=True)

                # Gradient step
                self.spectral_net.train()
                self.optimizer.zero_grad()

                Y, _, _ = self.spectral_net(X_grad, should_update_orth_weights=False)

                # Transform data using SiameseNet
                with torch.no_grad():
                    X_grad = siamese_net.single_forward(X_grad)
                    X_orth = siamese_net.single_forward(X_orth)

                W = self._get_affinity_matrix(X_grad)

                loss = self.criterion(W, Y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation step
            valid_loss = self.validate(valid_loader)
            self.scheduler.step(train_loss)

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

    def train(self, X: torch.Tensor | None = None, y: torch.Tensor | None = None):
        if X is None or y is None:
            raise ValueError("X and y are required for training")
        # Flatten the input tensor
        self.X = X.view(X.size(0), -1)
        self.y = y

        self.counter = 0
        self.criterion = SpectralNetLoss()

        self.spectral_net = SpectralNetModel(
            self.architecture, input_dim=self.X.shape[1]
        ).to(self.device)

        self.optimizer = optim.Adam(self.spectral_net.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        train_loader, ortho_loader, valid_loader, _ = self._get_data_loader(
            X=self.X, y=self.y
        )

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
                self.spectral_net(X_orth, should_update_orth_weights=True)

                # Gradient step
                self.spectral_net.train()
                self.optimizer.zero_grad()

                Y, _, _ = self.spectral_net(X_grad, should_update_orth_weights=False)

                W = self._get_affinity_matrix(X_grad)

                loss = self.criterion(W, Y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation step
            valid_loss = self.validate(valid_loader)
            self.scheduler.step(train_loss)

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

    def validate(self, valid_loader: DataLoader) -> float:
        valid_loss = 0.0
        self.spectral_net.eval()
        with torch.no_grad():
            for batch in valid_loader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)

                if self.is_sparse:
                    X = make_batch_for_sparse_grapsh(X)

                Y, _, _ = self.spectral_net(X, False)

                W = self._get_affinity_matrix(X)

                loss = self.criterion(W, Y)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        return valid_loss

    def predict(self, X: torch.Tensor, n_clusters) -> np.ndarray:
        X = X.view(X.size(0), -1)
        X = X.to(self.device)

        with torch.no_grad():
            self.embeddings_, _, _ = self.spectral_net(
                X, should_update_orth_weights=False
            )
            self.embeddings_ = self.embeddings_.detach().cpu().numpy()

        cluster_assignments = get_clusters_by_kmeans(self.embeddings_, n_clusters)
        return cluster_assignments
