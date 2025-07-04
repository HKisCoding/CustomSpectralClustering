import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.Config import Config
from utils.Process import EProjSimplex_new_matrix
from utils.Utils import affinity_to_adjacency, create_affinity_matrix

from .GraphConvolutionNetwork import GCN
from .SpectralNet import SpectralNetModel


class SCHOOL(nn.Module):
    def __init__(self, config: Config, orthonorm_weights: torch.Tensor | None = None):
        super().__init__()
        self.config = config

        self.spectral_net = SpectralNetModel(
            architecture=self.config.school.spectral_architecture,
            input_dim=self.config.school.feat_size,
            orthonorm_weights=orthonorm_weights,
        ).to(self.config.device)

        self.graph_encoder = GCN(
            channels_list=[
                self.config.school.feat_size,
                *self.config.school.gcn_architecture,
            ],
        ).to(self.config.device)

    def forward(
        self,
        x: torch.Tensor,
        x_orth: torch.Tensor,
        idx,
        alpha,
        beta,
        affinity_matrix,
        is_training: bool = True,
    ):
        if is_training:
            self.spectral_net(
                x=x_orth,
                semantic_out_dim=self.config.school.out_feat,
                should_update_orth_weights=True,
            )

        device = self.config.device or torch.device("cpu")
        # affinity_matrix = create_affinity_matrix(
        #     X=x,
        #     n_neighbors=self.config.school.n_neighbors,
        #     scale_k=self.config.school.k,
        #     device=device,
        # )
        self.init_graph = affinity_to_adjacency(affinity_matrix)

        Y, semantic_H, ortho_H = self.spectral_net(
            x=x,
            semantic_out_dim=self.config.school.out_feat,
            should_update_orth_weights=False,
        )

        num = x.shape[0]
        A = torch.zeros(num, num, device=device)
        idxa0 = idx[:, 1 : self.config.school.k + 1]
        dfi = torch.sqrt(torch.sum((Y.unsqueeze(1) - Y[idxa0]) ** 2, dim=2) + 1e-8).to(
            device
        )
        dxi = torch.sqrt(
            torch.sum((ortho_H.unsqueeze(1) - ortho_H[idxa0]) ** 2, dim=2) + 1e-8
        ).to(device)
        ad = -(dxi + beta * dfi) / (2 * alpha)
        A.scatter_(1, idxa0.to(device), EProjSimplex_new_matrix(ad))

        embs_hom = torch.mm(A, semantic_H)

        embs_graph = self.graph_encoder(x=x, adj_matrix=self.init_graph.to(device))

        return embs_hom, embs_graph, A, Y
