import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.Config import Config
from utils.Process import EProjSimplex_new_matrix
from utils.Utils import affinity_to_adjacency, create_affinity_matrix

from .GraphConvolutionNetwork import GCN
from .SpectralNet import SpectralNetModel


class SCHOOL(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.spectral_net = SpectralNetModel(
            architecture=self.config.spectral.architecture,
            input_dim=self.config.self_adjust_graph.feat_size,
        ).to(self.config.device)

        self.graph_encoder = GCN(
            in_channels=self.config.school.node_size,
            hidden_channels=self.config.school.gcn_hid_units,
            out_channels=self.config.school.hid_units,
        ).to(self.config.device)

    def forward(
        self,
        x: torch.Tensor,
        x_orth: torch.Tensor,
        beta: float,
        alpha: float,
        idx,
    ):
        self.spectral_net(
            x=x_orth,
            semantic_out_dim=self.config.self_adjust_graph.out_feat,
            should_update_orth_weights=True,
        )

        device = self.config.device or torch.device("cpu")
        affinty_matrix = create_affinity_matrix(
            X=x,
            n_neighbors=self.config.school.n_neighbors,
            scale_k=self.config.spectral.scale_k,
            device=device,
        )
        self.init_graph = affinity_to_adjacency(affinty_matrix)

        Y, semantic_H, ortho_H = self.spectral_net(
            x=x,
            semantic_out_dim=self.config.self_adjust_graph.out_feat,
            should_update_orth_weights=False,
        )

        num = x.shape[0]
        A = torch.zeros(num, num, device=device)
        idxa0 = idx[:, 1 : self.config.self_adjust_graph.k + 1]
        dfi = torch.sqrt(torch.sum((Y.unsqueeze(1) - Y[idxa0]) ** 2, dim=2)).to(device)
        dxi = torch.sqrt(
            torch.sum((ortho_H.unsqueeze(1) - ortho_H[idxa0]) ** 2, dim=2) + 1e-8
        ).to(device)
        ad = -(dxi + beta * dfi) / (2 * alpha)
        A.scatter_(1, idxa0.to(device), EProjSimplex_new_matrix(ad))
        embs_hom = torch.mm(A, semantic_H)

        embs_graph = self.graph_encoder(x=x, adj=self.init_graph)

        return embs_hom, embs_graph, A, Y
