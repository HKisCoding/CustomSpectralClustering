import torch
import torch.nn as nn


class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()

    def forward(
        self, W: torch.Tensor, Y: torch.Tensor, is_normalized: bool = False
    ) -> torch.Tensor:
        """
        This function computes the loss of the SpectralNet model.
        The loss is the rayleigh quotient of the Laplacian matrix obtained from W,
        and the orthonormalized output of the network.

        Args:
            W (torch.Tensor):               Affinity matrix
            Y (torch.Tensor):               Output of the network
            is_normalized (bool, optional): Whether to use the normalized Laplacian matrix or not.

        Returns:
            torch.Tensor: The loss
        """
        m = Y.size(0)
        if is_normalized:
            D = torch.sum(W, dim=1)
            Y = Y / torch.sqrt(D)[:, None]

        Dy = torch.cdist(Y, Y)
        loss = torch.sum(W * Dy.pow(2)) / (2 * m)

        return loss


class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps

    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  # [d, B]
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.0

    def forward(self, X):
        return -self.compute_discrimn_loss(X.T)


class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, eps=0.01, gamma=1):
        super(MaximalCodingRateReduction, self).__init__()
        self.eps = eps
        self.gamma = gamma

    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.0

    def compute_compress_loss(self, W, Pi):
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p, device=W.device).expand((k, p, p))
        trPi = Pi.sum(2) + 1e-8
        scale = (p / (trPi * self.eps)).view(k, 1, 1)

        W = W.view((1, p, m))
        log_det = torch.logdet(I + scale * W.mul(Pi).matmul(W.transpose(1, 2)))
        compress_loss = (trPi.squeeze() * log_det / (2 * m)).sum()
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        # This function support Y as label integer or membership probablity.
        if len(Y.shape) == 1:
            # if Y is a label vector
            if num_classes is None:
                num_classes = Y.max() + 1
            Pi = torch.zeros((num_classes, 1, Y.shape[0]), device=Y.device)
            for indx, label in enumerate(Y):
                Pi[label, 0, indx] = 1
        else:
            # if Y is a probility matrix
            if num_classes is None:
                num_classes = Y.shape[1]
            Pi = Y.T.reshape((num_classes, 1, -1))

        W = X.T
        discrimn_loss = self.compute_discrimn_loss(W)
        compress_loss = self.compute_compress_loss(W, Pi)

        total_loss = -discrimn_loss + self.gamma * compress_loss
        return total_loss, [discrimn_loss.item(), compress_loss.item()]


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(
        self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the contrastive loss between the two outputs of the siamese network.

        Parameters
        ----------
        output1 : torch.Tensor
            The first output of the siamese network.
        output2 : torch.Tensor
            The second output of the siamese network.
        label : torch.Tensor
            The label indicating whether the two outputs are similar (1) or not (0).

        Returns
        -------
        torch.Tensor
            The computed contrastive loss value.

        Notes
        -----
        This function takes the two outputs `output1` and `output2` of the siamese network,
        along with the corresponding `label` indicating whether the outputs are similar (1) or not (0).
        The contrastive loss is computed based on the Euclidean distance between the outputs and the label,
        and the computed loss value is returned.
        """

        euclidean = nn.functional.pairwise_distance(output1, output2)
        positive_distance = torch.pow(euclidean, 2)
        negative_distance = torch.pow(torch.clamp(self.margin - euclidean, min=0.0), 2)
        loss = torch.mean(
            (label * positive_distance) + ((1 - label) * negative_distance)
        )
        return loss
