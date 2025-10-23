import copy
import math
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.models.net.GraphAutoEncoder import AdaptiveGAE
from src.models.net.SCHOOL import GraphEncoderSchool
from utils.Config import Config
from utils.Loss import GAELoss, KLClusteringLoss, SpectralNetLoss
from utils.Metrics import run_evaluate_with_labels
from utils.Process import pairwise_distance
from utils.Utils import get_cluster_centroids, get_clusters_by_kmeans

from .BaseTrainer import BaseTrainer

INF = 1e-8


class SelfAdjustGraphEncoderTrainer(BaseTrainer):
    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config

        # Training config parameters
        self.lr = self.config.training.lr
        self.num_epoch = self.config.training.num_epoch
        self.weight_path = self.config.training.weight_path
        self.save_every = self.config.training.save_every

        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(self.weight_path, "weights"), exist_ok=True)

        # SelfAdjustGraph config parameters
        self.g_dim = self.config.self_adjust_graph.g_dim
        self.gamma = self.config.self_adjust_graph.gamma
        self.mu = self.config.self_adjust_graph.mu
        self.delta = self.config.self_adjust_graph.delta
        self.cluster = self.config.self_adjust_graph.cluster
        self.kl_alpha = self.config.self_adjust_graph.auxillary_loss_alpha
        self.auxillary_loss_kind = self.config.self_adjust_graph.auxillary_loss_kind
        self.theta = self.config.self_adjust_graph.theta

        self.feat_size = self.config.school.feat_size
        self.out_feat = self.config.school.out_feat
        self.k = self.config.school.k

        self.model = GraphEncoderSchool(self.config).to(self.config.device)
        self.adaptive_gae = AdaptiveGAE(
            channels_list=[
                self.config.school.feat_size,
                *self.config.school.gae_architecture,
            ],
        ).to(self.config.device)
        self.criterion = SpectralNetLoss()
        self.kl_loss = KLClusteringLoss(alpha=self.kl_alpha)
        self.gaeloss = GAELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            eps=1e-6,
            # weight_decay=0.0005,
            amsgrad=True,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.embedding = nn.Sequential(
            nn.Linear(
                self.out_feat,
                self.g_dim,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        ).to(self.config.device)

    def replace_nan_in_model(self):
        """Replace NaN values with 1e-8 in all model parameters."""
        for param in self.model.parameters():
            if param is not None:
                param.data = torch.nan_to_num(param.data, nan=1e-8)
        for param in self.embedding.parameters():
            if param is not None:
                param.data = torch.nan_to_num(param.data, nan=1e-8)

    def stratify_sampling(self, features: torch.Tensor, labels: torch.Tensor):
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Then perform the split as above
        X_train, X_val, y_train, y_val = train_test_split(
            features_np, labels_np, train_size=0.8, stratify=labels_np, random_state=42
        )

        # Convert back to tensors if needed
        X_train_tensor = torch.tensor(X_train).to(self.config.device)
        X_val_tensor = torch.tensor(X_val).to(self.config.device)
        y_train_tensor = torch.tensor(y_train).to(self.config.device)
        y_val_tensor = torch.tensor(y_val).to(self.config.device)
        return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor

    def compute_graph_parameters(self, X_grad: torch.Tensor):
        """Compute graph parameters from input features.

        Args:
            X_grad (torch.Tensor): Input features tensor

        Returns:
            tuple: (A, alpha, beta, idx) where:
                - A: Adjacency matrix
                - alpha: Mean of radius values
                - beta: Mean of radius values (same as alpha)
                - idx: Sorted indices of pairwise distances
        """
        X = X_grad.cpu()
        distX = pairwise_distance(X)
        # Sort the distances and get the sorted indices
        distX_sorted, idx = torch.sort(distX, dim=1)
        num = X_grad.shape[0]
        A = torch.zeros(num, num)
        rr = torch.zeros(num)
        for i in range(num):
            di = distX_sorted[i, 1 : self.k + 1]
            rr[i] = 0.5 * (self.k * di[self.k - 1] - torch.sum(di[: self.k]))
            id = idx[i, 1 : self.k + 1]
            A[i, id] = (di[self.k - 1] - di) / (
                self.k * di[self.k - 1]
                - torch.sum(di[: self.k])
                + torch.finfo(torch.float).eps
            )
        alpha = rr.mean()
        # r = 0

        beta = rr.mean()

        return A, alpha, beta, idx

    def auxillary_loss(self, Y: torch.Tensor, kind="entropy"):
        loss = 0
        if kind == "entropy":
            p_i = Y.sum(0).view(-1)
            p_i = (p_i + INF) / (p_i.sum() + INF)
            p_i = torch.abs(p_i)
            # The second term in Eq. (13): entropy loss
            loss = (
                math.log(p_i.size(0) + INF) + ((p_i + INF) * torch.log(p_i + INF)).sum()
            )
        elif kind == "KLdivergence":
            centroids = get_cluster_centroids(Y.detach().cpu().numpy(), self.cluster)
            centroids = torch.tensor(centroids, device=Y.device)
            loss = KLClusteringLoss(alpha=self.kl_alpha)(Y, centroids)
        return loss

    def compute_soft_assignments(self, embeddings, cluster_centers):
        """
        Compute soft assignment distribution Q using Student's t-distribution.

        Args:
            embeddings: tensor of shape (n_samples, embedding_dim) - spectral embedded points y_i
            cluster_centers: tensor of shape (n_clusters, embedding_dim) - cluster centers μ_j

        Returns:
            Q: soft assignment matrix of shape (n_samples, n_clusters)
        """
        # Compute squared distances between embeddings and cluster centers
        # ||y_i - μ_j||^2
        alpha = 1.0
        distances = torch.cdist(embeddings, cluster_centers, p=2) ** 2

        # Compute q_ij using Student's t-distribution kernel
        # q_ij = (1 + ||y_i - μ_j||^2 / α)^(-(α+1)/2)
        numerator = (1 + distances / alpha) ** (-(alpha + 1) / 2)

        # Normalize to get probability distribution (sum over j for each i)
        Q = numerator / torch.sum(numerator, dim=1, keepdim=True)

        return Q

    def train(
        self,
        features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        if features is None or labels is None:
            raise ValueError("Features and labels are required for training")

        # X_train, y_train, X_val, y_val = self.stratify_sampling(features, labels)
        # self.features_orth = X_train.to(self.config.device)
        # self.features = X_train.to(self.config.device)

        # X = self.features.cpu()
        # A, alpha, beta, idx = self.compute_graph_parameters(X)
        # X_grad = self.features
        # X_orth = self.features_orth[torch.randperm(self.features_orth.size(0))]
        self.features = features.view(features.size(0), -1).to(self.config.device)
        self.labels = labels.to(self.config.device)

        self.logger.info(f"Starting training for {self.num_epoch} epochs")
        self.logger.info(
            f"Training parameters: lr={self.lr}, gamma={self.gamma}, mu={self.mu}, delta={self.delta}"
        )

        train_loader, ortho_loader, val_loader, _ = self._get_data_loader(
            X=self.features, y=labels
        )
        start_time = time.time()
        pbar = tqdm(range(self.num_epoch), desc="Training")
        results = []
        best_train_loss = float("inf")
        best_spectral_loss = float("inf")
        for epoch in pbar:
            train_loss = 0
            epoch_spectral_loss = 0
            val_loss = 0
            epoch_node_consistency_loss = 0
            epoch_cluster_loss = 0
            self.model.train()
            epoch_start_time = time.time()
            for (X_grad, _), (X_orth, _) in zip(train_loader, ortho_loader):
                X_grad = X_grad.view(X_grad.size(0), -1)
                X_orth = X_orth.view(X_orth.size(0), -1)

                A, alpha, beta, idx = self.compute_graph_parameters(X_grad)

                embs_hom, A_updated, Y = self.model(X_grad, X_orth, idx, alpha, beta, A)

                A_construsted = self.adaptive_gae(A_updated, X_grad)

                construstion_loss = self.gaeloss(
                    A_construsted,
                    A_updated,
                    A,
                    self.adaptive_gae.embedding,
                    X_grad.shape[0],
                )
                embs_graph = self.adaptive_gae.embedding

                cluster_centers = get_cluster_centroids(
                    Y.detach().cpu().numpy(), self.cluster
                )
                cluster_centers = torch.tensor(cluster_centers, device=Y.device)
                Q = self.compute_soft_assignments(Y, cluster_centers)

                # # Replace NaN values in model parameters
                # self.replace_nan_in_model()

                spectral_loss = self.criterion(A_updated, Y)

                p_i = Q.sum(0).view(-1)
                p_i = (p_i + INF) / (p_i.sum() + INF)
                p_i = torch.abs(p_i)
                # The second term in Eq. (13): entropy loss
                entrpoy_loss = (
                    math.log(p_i.size(0) + INF)
                    + ((p_i + INF) * torch.log(p_i + INF)).sum()
                )
                spectral_loss = spectral_loss + self.gamma * entrpoy_loss

                embs_graph = self.embedding(embs_graph)
                embs_hom = self.embedding(embs_hom)

                #######################################################################
                # The first term in Eq. (15): invariance loss
                inter_c = embs_hom.T @ embs_graph.detach()
                inter_c = F.normalize(inter_c, p=2, dim=1)
                loss_inv = -torch.diagonal(inter_c).sum()

                # The second term in Eq. (15): uniformity loss
                # intra_c = (embs_hom).T @ (embs_hom).contiguous()
                # intra_c = torch.exp(F.normalize(intra_c, p=2, dim=1)).sum()
                # loss_uni = torch.log(intra_c).mean()

                # intra_c_2 = (embs_graph).T @ (embs_graph).contiguous()
                # intra_c_2 = torch.exp(F.normalize(intra_c_2, p=2, dim=1)).sum()
                # loss_uni += torch.log(intra_c_2).mean()
                intra_c = (embs_hom).T @ (embs_hom).contiguous() + (embs_graph).T @ (
                    embs_graph
                ).contiguous()
                intra_c = torch.exp(F.normalize(intra_c, p=2, dim=1)).sum()
                loss_uni = torch.log(intra_c)
                loss_consistency = loss_inv + self.mu * loss_uni

                # The second term in Eq. (13): cluster-level loss
                Y_hat = torch.argmax(Q, dim=1)
                avg_pooling_cluster_center = torch.stack(
                    [
                        torch.mean(embs_hom[Y_hat == i], dim=0)
                        for i in range(self.cluster)
                    ]
                )  # Shape: (num_clusters, embedding_dim)
                # Gather positive cluster centers
                # positive = avg_pooling_cluster_center[Y_hat]
                # # The first term in Eq. (11)
                # inter_c = positive.T @ embs_graph
                # inter_c = F.normalize(inter_c, p=2, dim=1)
                # loss_spe_inv = -torch.diagonal(inter_c).sum()

                loss_spe_inv = self.kl_loss(embs_graph, avg_pooling_cluster_center)

                loss = (
                    spectral_loss
                    + self.mu * loss_consistency
                    + self.delta * (loss_spe_inv)
                    + self.theta * construstion_loss
                )

                # Backward pass
                loss.backward()

                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                train_loss += loss.item()
                epoch_spectral_loss += spectral_loss.item()
                epoch_node_consistency_loss += loss_consistency.item()
                epoch_cluster_loss += loss_spe_inv.item()
                # Update progress bar with current metrics
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "spec_loss": f"{spectral_loss.item():.4f}",
                        "loss_consistency": f"{loss_consistency.item():.4f}",
                        "loss_spe_inv": f"{loss_spe_inv.item():.4f}",
                    }
                )

            train_loss /= len(train_loader)
            epoch_spectral_loss /= len(train_loader)
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            self.scheduler.step(train_loss)

            # Log detailed metrics every 10 epochs
            if (epoch + 1) % 10 == 0:
                # for X_val, _ in val_loader:
                #     X_val = X_val.view(X_val.size(0), -1)
                #     _, loss = self.validation(X_val)
                #     val_loss += loss
                # val_loss /= len(val_loader)
                # # _, loss = self.validation(X_val)
                # # val_loss = loss

                # self.scheduler.step(val_loss)

                self.logger.info(
                    f"Epoch [{epoch + 1}/{self.num_epoch}] "
                    f"Loss: {train_loss:.4f} "
                    f"Val Loss: {val_loss:.4f} "
                    f"Spectral Loss: {epoch_spectral_loss:.4f} "
                    f"Node Consistency Loss: {epoch_node_consistency_loss:.4f} "
                    f"Cluster Loss: {epoch_cluster_loss:.4f} "
                    f"Time: {epoch_time:.2f}s"
                )

            result = {
                "train_loss": train_loss,
                "spectral_loss": epoch_spectral_loss,
                "node_consistency_loss": epoch_node_consistency_loss,
                "cluster_loss": epoch_cluster_loss,
                "val_loss": val_loss,
            }
            results.append(result)

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        return results

    def validation(self, X_val: torch.Tensor):
        with torch.no_grad():
            self.model.eval()
            A_val, alpha, beta, idx = self.compute_graph_parameters(X_val)
            # Use the same features for both x and x_orth in validation
            _, _, A_val, Y = self.model(
                X_val, X_val, idx, alpha, beta, A_val, is_training=False
            )

            val_loss = self.criterion(A_val, Y)

        return Y, val_loss.item()

    def predict(
        self, X: torch.Tensor, n_clusters, use_weight: str | None = None
    ) -> np.ndarray:
        X = X.view(X.size(0), -1)
        X = X.to(self.config.device)

        if use_weight:
            # Load the best model
            best_model_path = os.path.join(self.weight_path, use_weight)
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.spectral_net.orthonorm_weights = checkpoint[
                    "orthonorm_weights"
                ]
            else:
                self.logger.warning("No best model found, using current model state")

        with torch.no_grad():
            self.embeddings_, _, _ = self.model.spectral_net(
                X, should_update_orth_weights=True
            )
            self.embeddings_ = self.embeddings_.detach().cpu().numpy()

        # cluster_assignments = get_clusters_by_kmeans(self.embeddings_, n_clusters)
        cluster_centers = get_cluster_centroids(self.embeddings_, n_clusters)
        cluster_centers = torch.tensor(cluster_centers, device=X.device)
        Y = torch.tensor(self.embeddings_, device=X.device)
        Q = self.compute_soft_assignments(Y, cluster_centers)
        cluster_assignments = torch.argmax(Q, dim=1).detach().cpu().numpy()

        return cluster_assignments
