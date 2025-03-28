import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer perceptron neural network with configurable dimensions and dropout."""

    def __init__(self, dim: list, dropprob=0.0):
        """Initialize MLP with given dimensions and dropout probability.

        Args:
            dim (list): List of integers specifying layer dimensions
            dropprob (float): Dropout probability (default: 0.0)
        """
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropprob)
        struc = []
        for i in range(len(dim)):
            struc.append(dim[i])
        for i in range(len(struc) - 1):
            self.net.append(nn.Linear(struc[i], struc[i + 1]))

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Output tensor after passing through all layers
        """
        for i in range(len(self.net) - 1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)

        # For the last layer
        y = self.net[-1](x)

        return y