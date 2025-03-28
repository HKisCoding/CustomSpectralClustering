import copy
import math
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.models.layers import FullyConnectedLayer
from src.models.net.SCHOOL import SCHOOL
from utils.Config import Config
from utils.Loss import SpectralClusterLoss
from utils.Process import pairwise_distance

from .BaseTrainer import BaseTrainer

INF = 1e-8


class SelfAdjustGraphTrainer(BaseTrainer):
    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config

        # Training config parameters
        self.lr = self.config.training.lr
        self.num_epoch = self.config.training.num_epoch
        self.weight_path = self.config.training.weight_path
        self.save_every = self.config.training.save_every

        # SelfAdjustGraph config parameters
        self.hid_unit = self.config.self_adjust_graph.hid_unit
        self.feat_size = self.config.self_adjust_graph.feat_size
        self.out_feat = self.config.self_adjust_graph.out_feat
        self.g_dim = self.config.self_adjust_graph.g_dim
        self.k = self.config.self_adjust_graph.k
        self.gamma = self.config.self_adjust_graph.gamma
        self.mu = self.config.self_adjust_graph.mu
        self.delta = self.config.self_adjust_graph.delta
        self.eta = self.config.self_adjust_graph.eta
        self.cluster = self.config.self_adjust_graph.cluster

        # Model components
        self.fc = FullyConnectedLayer(
            in_ft=self.hid_unit + self.feat_size,
            out_ft=self.out_feat,
        )

        self.model = SCHOOL(self.config).to(self.config.device)
        self.criterion = SpectralClusterLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.embedding = nn.Sequential(
            nn.Linear(
                self.out_feat,
                self.g_dim,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        ).to(self.config.device)

    def train(
        self,
        features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        if features is None or labels is None:
            raise ValueError("Features and labels are required for training")
        self.features_orth = features.to(self.config.device)
        self.features = features.to(self.config.device)

        X = self.features.cpu()

        distX = pairwise_distance(X)
        num_nodes = self.features.shape[0]
        distX_sorted, idx = torch.sort(distX, dim=1)

        A = torch.zeros(num_nodes, num_nodes)

        rr = torch.zeros(num_nodes)
        for i in range(num_nodes):
            di = distX_sorted[i, 1 : self.k + 1]
            rr[i] = 0.5 * (self.k * di[self.k - 1] - torch.sum(di[: self.k]))
            id = idx[i, 1 : self.k + 1]
            A[i, id] = (di[self.k - 1] - di) / (
                self.k * di[self.k - 1]
                - torch.sum(di[: self.k])
                + torch.finfo(torch.float).eps
            )
        alpha = rr.mean()
        beta = rr.mean()

        self.logger.info(f"Starting training for {self.num_epoch} epochs")
        self.logger.info(
            f"Training parameters: lr={self.lr}, gamma={self.gamma}, mu={self.mu}, delta={self.delta}"
        )

        train_loader, val_loader, _ = self._get_data_loader(X=self.features, y=labels)
        start_time = time.time()
        pbar = tqdm(range(self.num_epoch), desc="Training")
        for epoch in pbar:
            epoch_start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()

            for X_grad, _ in train_loader:
                X_orth = copy.deepcopy(X_grad)
                X_grad = X_grad.to(self.config.device)
                X_orth = X_orth.to(self.config.device)

                embs_hom, embs_graph, A, Y = self.model(
                    X_grad, X_orth, A, beta, alpha, idx
                )

                spectral_loss = self.criterion(A, Y)

                p_i = Y.sum(0).view(-1)
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
                intra_c = (embs_hom).T @ (embs_hom).contiguous()
                intra_c = torch.exp(F.normalize(intra_c, p=2, dim=1)).sum()
                loss_uni = torch.log(intra_c).mean()

                intra_c_2 = (embs_graph).T @ (embs_graph).contiguous()
                intra_c_2 = torch.exp(F.normalize(intra_c_2, p=2, dim=1)).sum()
                loss_uni += torch.log(intra_c_2).mean()
                loss_consistency = loss_inv + self.mu * loss_uni

                # The second term in Eq. (13): cluster-level loss
                Y_hat = torch.argmax(Y, dim=1)
                cluster_center = torch.stack(
                    [
                        torch.mean(embs_graph[Y_hat == i], dim=0)
                        for i in range(self.cluster)
                    ]
                )  # Shape: (num_clusters, embedding_dim)
                # Gather positive cluster centers
                positive = cluster_center[Y_hat]
                # The first term in Eq. (11)
                inter_c = positive.T @ embs_hom
                inter_c = F.normalize(inter_c, p=2, dim=1)
                loss_spe_inv = -torch.diagonal(inter_c).sum()

                loss = (
                    spectral_loss
                    + self.mu * loss_consistency
                    + self.delta * (loss_spe_inv)
                )

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time

                # Update progress bar with current metrics
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "spec_loss": f"{spectral_loss.item():.4f}",
                        "time": f"{epoch_time:.2f}s",
                    }
                )

                # Log detailed metrics every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.logger.info(
                        f"Epoch [{epoch + 1}/{self.num_epoch}] "
                        f"Loss: {loss.item():.4f} "
                        f"Spectral Loss: {spectral_loss.item():.4f} "
                        f"Consistency Loss: {loss_consistency.item():.4f} "
                        f"Cluster Loss: {loss_spe_inv.item():.4f} "
                        f"Time: {epoch_time:.2f}s"
                    )

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
