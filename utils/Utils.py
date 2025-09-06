import os

import numpy as np
import torch
from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors


def save_state(model_dir, *entries, filename="losses.csv"):
    """Save entries to csv. Entries is list of numbers."""
    csv_path = os.path.join(model_dir, filename)
    assert os.path.exists(csv_path), "CSV file is missing in project directory."
    with open(csv_path, "a") as f:
        f.write("\n" + ",".join(map(str, entries)))


def save_ckpt(model_dir, net, optimizer, scheduler, epoch):
    """Save PyTorch checkpoint to ./checkpoints/ directory in model directory."""
    save_dict = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(
        save_dict,
        os.path.join(model_dir, "checkpoints", "model-epoch{}.pt".format(epoch)),
    )


def calculate_cost_matrix(C: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Calculates the cost matrix for the Munkres algorithm.

    Parameters
    ----------
    C : np.ndarray
        Confusion matrix.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Cost matrix.
    """

    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices: np.ndarray) -> np.ndarray:
    """
    Gets the cluster labels from their indices.

    Parameters
    ----------
    indices : np.ndarray
        Indices of the clusters.

    Returns
    -------
    np.ndarray
        Cluster labels.
    """

    num_clusters = len(indices)
    cluster_labels = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_nearest_neighbors(
    X: torch.Tensor, Y: torch.Tensor = None, k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the distances and the indices of the k nearest neighbors of each data point.

    Parameters
    ----------
    X : torch.Tensor
        Batch of data points.
    Y : torch.Tensor, optional
        Defaults to None.
    k : int, optional
        Number of nearest neighbors to calculate. Defaults to 3.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Distances and indices of each data point.
    """
    if Y is None:
        Y = X
    if len(X) < k:
        k = len(X)
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    Dis, Ids = nbrs.kneighbors(X)
    return Dis, Ids


def compute_scale(
    Dis: np.ndarray, k: int = 2, med: bool = True, is_local: bool = True
) -> np.ndarray:
    """
    Computes the scale for the Gaussian similarity function.

    Parameters
    ----------
    Dis : np.ndarray
        Distances of the k nearest neighbors of each data point.
    k : int, optional
        Number of nearest neighbors for the scale calculation. Relevant for global scale only.
    med : bool, optional
        Scale calculation method. Can be calculated by the median distance from a data point to its neighbors,
        or by the maximum distance. Defaults to True.
    is_local : bool, optional
        Local distance (different for each data point), or global distance. Defaults to True.

    Returns
    -------
    np.ndarray
        Scale (global or local).
    """

    if is_local:
        if not med:
            scale = np.max(Dis, axis=1)
        else:
            scale = np.median(Dis, axis=1)
    else:
        if not med:
            scale = np.max(Dis[:, k - 1])
        else:
            scale = np.median(Dis[:, k - 1])
    return scale


def get_gaussian_kernel(
    D: torch.Tensor, scale, Ids: np.ndarray, device: torch.device, is_local: bool = True
) -> torch.Tensor:
    """
    Computes the Gaussian similarity function according to a given distance matrix D and a given scale.

    Parameters
    ----------
    D : torch.Tensor
        Distance matrix.
    scale :
        Scale.
    Ids : np.ndarray
        Indices of the k nearest neighbors of each sample.
    device : torch.device
        Defaults to torch.device("cpu").
    is_local : bool, optional
        Determines whether the given scale is global or local. Defaults to True.

    Returns
    -------
    torch.Tensor
        Matrix W with Gaussian similarities.
    """

    if not is_local:
        # global scale
        W = torch.exp(-torch.pow(D, 2) / (scale**2))
    else:
        # local scales
        W = torch.exp(
            -torch.pow(D, 2).to(device)
            / (torch.tensor(scale).float().to(device).clamp_min(1e-7) ** 2)
        )
    if Ids is not None:
        n, k = Ids.shape
        mask = torch.zeros([n, n]).to(device=device)
        for i in range(len(Ids)):
            mask[i, Ids[i]] = 1
        W = W * mask
    sym_W = (W + torch.t(W)) / 2.0
    return sym_W


def make_batch_for_sparse_grapsh(batch_x: torch.Tensor) -> torch.Tensor:
    """
    Computes a new batch of data points from the given batch (batch_x)
    in case that the graph-laplacian obtained from the given batch is sparse.
    The new batch is computed based on the nearest neighbors of 0.25
    of the given batch.

    Parameters
    ----------
    batch_x : torch.Tensor
        Batch of data points.

    Returns
    -------
    torch.Tensor
        New batch of data points.
    """

    batch_size = batch_x.shape[0]
    batch_size //= 5
    new_batch_x = batch_x[:batch_size]
    batch_x = new_batch_x
    n_neighbors = 5

    u = AnnoyIndex(batch_x[0].shape[0], "euclidean")
    u.load("ann_index.ann")
    for x in batch_x:
        x = x.detach().cpu().numpy()
        nn_indices = u.get_nns_by_vector(x, n_neighbors)
        nn_tensors = [u.get_item_vector(i) for i in nn_indices[1:]]
        nn_tensors = torch.tensor(nn_tensors, device=batch_x.device)
        new_batch_x = torch.cat((new_batch_x, nn_tensors))

    return new_batch_x


def create_affinity_matrix(
    X: torch.Tensor, n_neighbors: int, scale_k: int, device: torch.device
):
    """
    Computes the affinity matrix for the given data points.

    Parameters
    ----------
    X : torch.Tensor
        Data points.

    Returns
    -------
    torch.Tensor
        Affinity matrix.
    """
    Dx = torch.cdist(X, X)
    Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
    scale = compute_scale(Dis, k=scale_k, is_local=True)
    W = get_gaussian_kernel(Dx, scale, indices, device=device, is_local=True)
    return W


def affinity_to_adjacency(
    affinity_matrix: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Convert affinity matrix to adjacency matrix using thresholding.

    Parameters
    ----------
    affinity_matrix : torch.Tensor
        Input affinity matrix
    threshold : float, optional
        Threshold value to determine edges (default: 0.5)

    Returns
    -------
    torch.Tensor
        Binary adjacency matrix
    """
    # Create binary adjacency matrix based on threshold
    adjacency_matrix = (affinity_matrix > threshold).float()

    # Ensure diagonal is zero (no self-loops)
    adjacency_matrix.fill_diagonal_(0)

    # Make the matrix symmetric
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.t()) / 2

    return adjacency_matrix


def get_clusters_by_kmeans(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Performs k-means clustering on the spectral-embedding space.

    Parameters
    ----------
    embeddings : np.ndarray
        The spectral-embedding space.

    Returns
    -------
    np.ndarray
        The cluster assignments for the given data.
    """
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(embeddings)
    cluster_assignments = kmeans.predict(embeddings)
    return cluster_assignments


def get_cluster_centroids(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(embeddings)
    centroids = kmeans.cluster_centers_
    return centroids
