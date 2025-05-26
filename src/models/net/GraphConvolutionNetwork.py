import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        # First Graph Convolutional Layer
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Batch normalization after first conv
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        # Second Graph Convolutional Layer
        self.conv2 = GCNConv(hidden_channels, out_channels)
        # Batch normalization after second conv
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, adj_matrix):
        edge_index, edge_weight = dense_to_sparse(adj_matrix)
        # First Conv layer with batch s
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)

        # Second Conv layer with batch normalization
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)

        return x
