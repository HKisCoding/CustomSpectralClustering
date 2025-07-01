import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


class GCN(nn.Module):
    def __init__(self, channels_list):
        super(GCN, self).__init__()
        """
        Initialize GCN with a list of channel sizes.

        Args:
            channels_list (list): List of integers representing channel sizes.
                                e.g., [input_dim, hidden_dim1, hidden_dim2, ..., output_dim]
        """
        if len(channels_list) < 2:
            raise ValueError(
                "channels_list must have at least 2 elements (input and output dimensions)"
            )

        self.channels_list = channels_list
        self.num_layers = len(channels_list) - 1

        # Create layers dynamically based on channels_list
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(self.num_layers):
            in_channels = channels_list[i]
            out_channels = channels_list[i + 1]

            # Add Graph Convolutional Layer
            self.conv_layers.append(GCNConv(in_channels, out_channels))
            self.bn_layers.append(nn.BatchNorm1d(out_channels))

    def forward(self, x, adj_matrix):
        edge_index, edge_weight = dense_to_sparse(adj_matrix)

        # Apply layers
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index, edge_weight)
            x = self.bn_layers[i](x)
            # Apply batch normalization and activation (except for the last layer)
            if i < self.num_layers - 1:
                x = F.relu(x)

        return x
