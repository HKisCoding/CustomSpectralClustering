import numpy as np
import torch
import torch.nn as nn


class SpectralNetModel(nn.Module):
    def __init__(
        self,
        architecture: list,
        input_dim: int,
        orthonorm_weights: torch.Tensor | None = None,
    ):
        super(SpectralNetModel, self).__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.orthonorm_weights = orthonorm_weights

        current_dim = self.input_dim
        for i, layer in enumerate(self.architecture):
            next_dim = layer
            if i == len(self.architecture) - 1:
                # Last layer -> number of cluster
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.Tanh())
                )
            else:
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.LeakyReLU())
                )
                current_dim = next_dim

    def orthonormalize(self, input: torch.Tensor) -> torch.Tensor:
        """
        Take the output of model and apply the orthonormalization using Cholesky decomposition

        Args:
            input (torch.Tensor): output of gradient model.

        Returns:
            torch.Tensor: The orthonormalize weight after decomposition.
        """

        m = input.shape[0]
        _, R = torch.linalg.qr(input)
        orthonorm_weights = np.sqrt(m) * torch.linalg.inv(R)
        return orthonorm_weights

    def forward(
        self,
        x: torch.Tensor,
        semantic_out_dim: int | None = None,
        should_update_orth_weights: bool = True,
    ):
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: output tensor
        """
        i = 0
        semantic_H = torch.zeros(x.shape, device=x.device)
        for layer in self.layers:
            x = layer(x)
            if self.architecture[i] == semantic_out_dim:
                semantic_H = x
            i += 1

        Y_tilde = x
        Y2_tilde = semantic_H
        if should_update_orth_weights:
            self.orthonorm_weights = self.orthonormalize(Y_tilde)
            self.orthonorm_weights_2 = (
                self.orthonormalize(Y2_tilde) if semantic_out_dim else semantic_H.T
            )
        Y = Y_tilde @ self.orthonorm_weights
        if not torch.all(semantic_H == 0):
            ortho_H = Y2_tilde @ self.orthonorm_weights_2
        else:
            ortho_H = None
        return Y, semantic_H, ortho_H
