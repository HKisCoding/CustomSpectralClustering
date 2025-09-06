import torch
import torch.nn as nn
import torch.nn.functional as F


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


class KLClusteringLoss(nn.Module):
    """
    Kullback-Leibler divergence loss for clustering based on soft assignments.

    This implements the clustering loss from Deep Embedded Clustering (DEC) where:
    - Q is the soft assignment distribution
    - P is the target distribution computed by sharpening Q
    """

    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: degrees of freedom parameter for Student's t-distribution (default: 1.0)
        """
        super(KLClusteringLoss, self).__init__()
        self.alpha = alpha

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
        distances = torch.cdist(embeddings, cluster_centers, p=2) ** 2

        # Compute q_ij using Student's t-distribution kernel
        # q_ij = (1 + ||y_i - μ_j||^2 / α)^(-(α+1)/2)
        numerator = (1 + distances / self.alpha) ** (-(self.alpha + 1) / 2)

        # Normalize to get probability distribution (sum over j for each i)
        Q = numerator / torch.sum(numerator, dim=1, keepdim=True)

        return Q

    def compute_target_distribution(self, Q):
        """
        Compute target distribution P by sharpening Q.

        Args:
            Q: soft assignment matrix of shape (n_samples, n_clusters)

        Returns:
            P: target distribution of shape (n_samples, n_clusters)
        """
        # Square the soft assignments and normalize by cluster frequency
        # p_ij = q_ij^2 / Σ_i q_ij

        f_j = torch.sum(Q, dim=0, keepdim=True)
        squared_Q = Q**2 / f_j

        # # Avoid division by zero
        # cluster_frequencies = torch.clamp(cluster_frequencies, min=1e-10)

        # Normalize by cluster frequencies
        cluster_frequency = squared_Q / f_j

        # Normalize to get probability distribution (sum over j for each i)
        P = cluster_frequency / torch.sum(cluster_frequency, dim=1, keepdim=True)

        return P

    def kl_divergence(self, P, Q):
        """
        Compute KL divergence KL(P || Q).

        Args:
            P: target distribution of shape (n_samples, n_clusters)
            Q: soft assignment distribution of shape (n_samples, n_clusters)

        Returns:
            kl_loss: scalar tensor representing the KL divergence loss
        """
        # Add small epsilon to avoid log(0)
        # epsilon = 1e-10
        # Q_safe = torch.clamp(Q, min=epsilon)

        # Compute KL(P || Q) = Σ_i Σ_j p_ij * log(p_ij / q_ij)
        kl_loss = torch.sum(P * torch.log(P / Q))

        return kl_loss

    def forward(self, embeddings, cluster_centers):
        """
        Forward pass computing the complete clustering loss.

        Args:
            embeddings: tensor of shape (n_samples, embedding_dim)
            cluster_centers: tensor of shape (n_clusters, embedding_dim)

        Returns:
            loss: scalar tensor representing the clustering loss
            Q: soft assignment distribution (for auxiliary outputs)
            P: target distribution (for auxiliary outputs)
        """
        # Compute soft assignments Q
        Q = self.compute_soft_assignments(embeddings, cluster_centers)

        # Compute target distribution P
        P = self.compute_target_distribution(Q)

        # Compute KL divergence loss
        loss = self.kl_divergence(P, Q)

        return loss


class MSELoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_true_batch, y_pred_batch, hidden_units):
        """
        Total loss function of DSC

        Args:
            y_true_batch: target matrix Y (torch tensor)
            y_pred_batch: outputs of the encoder (torch tensor)
            hidden_units: number of hidden units

        Returns:
            loss: MSE loss between target and prediction
        """
        # Extract only the embedding part (first hidden_units dimensions)
        y_pred_batch = y_pred_batch[:, :hidden_units]
        loss = F.mse_loss(y_true_batch, y_pred_batch)
        return loss
