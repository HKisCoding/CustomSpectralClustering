import numpy as np
import torch
import torch.nn as nn


class AdaptiveGAE(nn.Module):
    def __init__(
        self,
        channels_list: list,
    ):
        super(AdaptiveGAE, self).__init__()
        """
        Initialize AdaptiveGAE with a list of channel sizes.

        Args:
            channels_list (list): List of integers representing channel sizes.
                                e.g., [input_dim, hidden_dim1, hidden_dim2, ..., output_dim]
        """
        if len(channels_list) < 2:
            raise ValueError(
                "channels_list must have at least 2 elements (input and output dimensions)"
            )

        self.channels_list = channels_list

        self.W1 = self._get_weight_initial([channels_list[0], channels_list[1]])
        self.W2 = self._get_weight_initial([channels_list[1], channels_list[2]])

    def _get_weight_initial(self, shape):
        bound = np.sqrt(6.0 / (shape[0] + shape[1]))
        ini = torch.rand(shape) * 2 * bound - bound
        return torch.nn.Parameter(ini, requires_grad=True)

    def get_embedding(self, W: torch.Tensor, X: torch.Tensor):
        D = torch.sum(W, dim=1)
        D_sqrt_inv = torch.diag(torch.pow(D + 1e-8, -0.5))
        L = D_sqrt_inv.mm(D - W).mm(D_sqrt_inv)
        embedding = L.mm(X.matmul(self.W1))
        embedding = torch.nn.functional.relu(embedding)
        return embedding, L

    def forward(self, W: torch.Tensor, X: torch.Tensor):
        """
        Forward pass of AdaptiveGAE.

        Args:
            W: Affinity matrix.
            X: Input features.

        Returns:
            recons_w: Reconstructed affinity matrix.
        """
        embedding, L = self.get_embedding(W, X)
        # sparse
        self.embedding = L.mm(embedding.matmul(self.W2))
        distances = distance(self.embedding.t(), self.embedding.t())
        softmax = torch.nn.Softmax(dim=1)
        recons_w = softmax(-distances)
        return recons_w + 10**-10


def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result
