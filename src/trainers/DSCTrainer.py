import os.path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from src.models.net.AE_conv import model_conv, train_base
from utils.Loss import MSELoss
from utils.Metrics import run_evaluate_with_labels
from utils.Utils import create_affinity_matrix, get_clusters_by_kmeans


def sorted_eig(X, device="cpu"):
    """
    Compute eigenvalues and eigenvectors of a matrix

    Args:
        X: Input matrix (numpy array or torch tensor)
        device: Device for computation ('cpu' or 'cuda')

    Returns:
        e_vals: Eigenvalues
        e_vecs: Eigenvectors
    """
    # Convert to torch tensor if numpy array
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float().to(device)
    elif not isinstance(X, torch.Tensor):
        X = torch.tensor(X).float().to(device)
    else:
        X = X.to(device)

    # Compute eigenvalues and eigenvectors using torch
    e_vals, e_vecs = torch.linalg.eigh(X)

    # Convert back to numpy for compatibility with existing code
    return e_vals.cpu().numpy(), e_vecs.cpu().numpy()


def transform_matrix_V(H_pi, U, n_clusters, assignment, device: torch.device):
    """
    Construct Orthonormal transformation matrix V

    Args:
        H_pi: autoencoder embeddings (numpy array or torch tensor)
        U: cluster centroids (Kmeans) (numpy array or torch tensor)
        n_clusters: number of clusters
        assignment: cluster indicator (numpy array)
        device: device for computation

    Returns:
        Evals: eigenvalues
        V: eigenvectors (called Orthonormal transformation matrix V)
    """
    # Convert inputs to torch tensors if needed
    if isinstance(H_pi, np.ndarray):
        H_pi = torch.from_numpy(H_pi).float().to(device)
    else:
        H_pi = H_pi.to(device)

    if isinstance(U, np.ndarray):
        U = torch.from_numpy(U).float().to(device)
    else:
        U = U.to(device)

    if isinstance(assignment, np.ndarray):
        assignment = torch.from_numpy(assignment).to(device)
    else:
        assignment = assignment.to(device)

    S_i = []
    for i in range(n_clusters):
        # Get points belonging to cluster i
        cluster_mask = assignment == i
        cluster_points = H_pi[cluster_mask]

        if cluster_points.size(0) > 0:  # Check if cluster is not empty
            # Compute deviation from centroid
            temp = cluster_points - U[i]
            # Compute covariance matrix: temp^T * temp
            temp = torch.matmul(temp.t(), temp)
            S_i.append(temp)
        else:
            # Handle empty cluster case
            temp = torch.zeros_like(torch.matmul(H_pi[:1].t(), H_pi[:1]))
            S_i.append(temp)

    # Stack and sum all covariance matrices
    S_i = torch.stack(S_i)
    S = torch.sum(S_i, dim=0)

    # Compute eigenvalues and eigenvectors
    Evals, V = sorted_eig(S.cpu().numpy(), device="cpu")  # Convert to numpy for eigh

    return Evals, V.astype(np.float32)


class DSC_trainer(object):
    """
    PyTorch implementation of DSC (Deep Subspace Clustering) class
    """

    def __init__(self, cfg, device: torch.device):
        self.model_type = cfg.MODEL_TYPE
        self.cfg = cfg
        self.device = device
        self.model_has_decoder = True
        self.model = model_conv(cfg, load_weights=False)
        self.model.to(self.device)

        self.hidden_units = self.cfg.DSCConfig.hidden_units
        self.batch_size = self.cfg.DSCConfig.batch_size
        self.n_neighbors = self.cfg.DSCConfig.n_neighbors
        self.scale_k = self.cfg.DSCConfig.scale_k

        self.ae_batch_size = self.cfg.AEConvConfig.batch_size
        self.ae_epochs = self.cfg.AEConvConfig.epochs
        self.ae_weight_path = self.cfg.AEConvConfig.weight_path

    def load_reconstruction_weights(self, cfg, path=None):
        """Load pre-trained autoencoder weights"""
        self.cfg = cfg if cfg else self.cfg
        if not path:
            path = cfg.AEConvConfig.weight_path
        if not os.path.exists(path):
            raise Exception("no weights available")

        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Loaded reconstruction weights from {path}")

    def get_AE_embeddings(self, x):
        """
        Get autoencoder embeddings from input data

        Args:
            x: Input data (numpy array or torch tensor)

        Returns:
            embeddings: Normalized embeddings from the model
        """
        self.model.eval()

        # Convert to tensor if numpy array
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Ensure correct device
        x = x.to(self.device)

        with torch.no_grad():
            if self.model_has_decoder:
                # If model has decoder, get full output and extract embeddings
                output, embeddings, _ = self.model(x)
                return embeddings.cpu().numpy()
            else:
                # If model is encoder-only, get embeddings directly
                embeddings = self.model(x)
                return embeddings.cpu().numpy()

    def train_reconstruction(
        self, x: torch.Tensor, epoch: torch.Tensor, dataset=None, cfg=None, y=None
    ):
        """
        Train the reconstruction (autoencoder) part of the model

        Args:
            x: Input data
            epoch: Number of training epochs
            dataset: PyTorch dataset (optional)
            cfg: Configuration object
            y: Labels (not used in reconstruction training)
        """
        self.cfg = cfg if cfg else self.cfg
        batchsize = self.cfg.AEConvConfig.batch_size

        if not epoch:
            epoch = self.cfg.AEConvConfig.epochs

        # Prepare dataloader
        if dataset:
            dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
        else:
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()

            # Create dataset for autoencoder training (input as both x and target)
            dataset = TensorDataset(x, x)
            dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

        # Train the base model
        train_base(self.model, dataloader, self.cfg, epoch, device=self.device)

        # Reload weights after training
        self.load_reconstruction_weights(self.cfg)

    def train(self, x, n_clusters):
        device = self.device
        # Convert inputs to appropriate formats
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x = x.to(device)

        # Configuration parameters

        # Training variables
        ite_PI = 0
        a_PI = 0
        n_skip = 0
        n_iter = self.cfg.DSCConfig.n_iter
        assignment = np.array([-1] * len(x))
        optimizer = torch.optim.Adam(self.model.parameters())

        # Main training loop
        for step in range(20):
            print(f"iter {step + 1}:", end="\t|\t")

            # Get autoencoder embeddings
            self.model.eval()
            with torch.no_grad():
                H = self.model(x)
                if isinstance(H, tuple):  # If model returns multiple outputs
                    H = H[0]  # Take first output (embeddings)
                H = H[:, : self.hidden_units].cpu().numpy()

            H_pi = H.copy()

            # Construct normalized affinity matrix W
            W = create_affinity_matrix(
                torch.Tensor(H_pi),
                n_neighbors=self.n_neighbors,
                scale_k=self.scale_k,
                device=self.device,
            )

            # Power iteration
            v = 0
            for ite_PI in range(15):
                H_pi_old = np.array(H_pi)
                v_old = np.array(v) if isinstance(v, np.ndarray) else v
                H_pi = W.dot(H_pi)
                v = np.mean(H_pi - H_pi_old, -1)
                a_PI = np.linalg.norm(v - v_old, ord=np.inf)
                if a_PI <= 1e-3:
                    break

            print(
                f"PI iterations={ite_PI}, acceleration={np.round(a_PI, 5)}", end="\t|\t"
            )

            # Cluster on power iteration embeddings H_pi (called Z in paper)
            kmeans = KMeans(n_clusters=n_clusters, n_init=50).fit(H_pi)
            assignment = kmeans.predict(H_pi)
            U_pi = kmeans.cluster_centers_

            # Construct Orthonormal transformation matrix V
            _, V = transform_matrix_V(
                H_pi, U_pi, n_clusters, assignment, device=self.device
            )

            # Transform autoencoder embeddings into new space
            H_v = H_pi @ V
            U_v = U_pi @ V

            # Construct training label (target matrix Y in paper)
            y_true = np.array(H_v)
            if step != 0:
                # Replace last dimension with cluster centroids (Equation 16 in paper)
                y_true[:, -1] = U_v[assignment][:, -1]
            y_true = y_true @ V.T

            # Convert to tensors
            y_true_tensor = torch.from_numpy(y_true).float().to(device)

            if step == 0:
                # Initial training (simplified version)
                dataset = TensorDataset(x, y_true_tensor)
                dataloader = DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=True
                )

                self.model.train()
                for epoch in range(10):
                    epoch_loss = 0.0
                    for x_batch, y_batch in dataloader:
                        optimizer.zero_grad()
                        y_pred = self.model(x_batch)
                        if isinstance(y_pred, tuple):
                            y_pred = y_pred[0]  # Take embeddings if multiple outputs
                        loss = MSELoss()(y_batch, y_pred, self.hidden_units)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                continue

            # Create dataset with skipping mechanism
            # PyTorch equivalent of TF's skip().batch().repeat().take()
            dataset = TensorDataset(x, y_true_tensor)

            # Create custom sampler for skipping
            indices = list(range(len(x)))
            if n_skip > 0:
                indices = indices[n_skip:] + indices[:n_skip]

            # Take only n_iter batches
            sampled_indices = []
            for i in range(n_iter):
                batch_start = (i * self.batch_size) % len(indices)
                batch_end = min(batch_start + self.batch_size, len(indices))
                sampled_indices.extend(indices[batch_start:batch_end])
                if batch_end < batch_start + self.batch_size:
                    # Wrap around
                    remaining = self.batch_size - (batch_end - batch_start)
                    sampled_indices.extend(indices[:remaining])

            # Update skip counter
            n_skip = (n_skip + n_iter * self.batch_size) % len(x)

            # Train on selected samples
            self.model.train()
            for i in range(0, len(sampled_indices), self.batch_size):
                batch_indices = sampled_indices[i : i + self.batch_size]
                x_batch = x[batch_indices]
                y_batch = y_true_tensor[batch_indices]

                optimizer.zero_grad()
                y_pred = self.model(x_batch)
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]  # Take embeddings if multiple outputs
                loss = torch.nn.functional.mse_loss(y_batch, y_pred)
                loss.backward()
                optimizer.step()

        return assignment

    def _create_encoder_only_model(self):
        """
        Create an encoder-only model from the full autoencoder
        This extracts just the encoder part up to the embedding layer
        """

        class EncoderOnly(nn.Module):
            def __init__(self, full_model):
                super(EncoderOnly, self).__init__()
                self.conv1 = full_model.conv1
                self.conv2 = full_model.conv2
                self.conv3 = full_model.conv3
                self.embed_dense = full_model.embed_dense

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                x = x.view(x.size(0), -1)
                h = self.embed_dense(x)
                # L2 normalization
                h_norm = torch.nn.functional.normalize(h, p=2, dim=-1)
                return h_norm

        encoder_model = EncoderOnly(self.model)
        encoder_model.to(self.device)
        return encoder_model

    def evaluate(self, x, y):
        """
        Evaluate clustering performance

        Args:
            x: Input data
            y: Ground truth labels

        Returns:
            acc: Clustering accuracy
            nmi: Normalized mutual information
        """
        n_clusters = len(np.unique(y))

        # Get embeddings
        self.model.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(self.device)

        with torch.no_grad():
            H = self.model(x).cpu().numpy()

        cluster_assignments = get_clusters_by_kmeans(H, n_clusters)

        result = run_evaluate_with_labels(
            cluster_assignments=cluster_assignments, y=y, n_clusters=n_clusters
        )
        return result
