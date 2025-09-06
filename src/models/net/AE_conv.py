import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# from utils.utils import *


class ConvAutoencoder(nn.Module):
    def __init__(self, input_shape, hidden_units):
        super(ConvAutoencoder, self).__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units

        # Calculate padding for the third conv layer
        self.pad3 = "same" if input_shape[0] % 8 == 0 else "valid"

        # Encoder layers
        filters = [32, 64, 128, hidden_units]
        self.conv1 = nn.Conv2d(
            input_shape[2], filters[0], kernel_size=5, stride=2, padding=2
        )
        self.conv2 = nn.Conv2d(
            filters[0], filters[1], kernel_size=5, stride=2, padding=2
        )

        if self.pad3 == "same":
            self.conv3 = nn.Conv2d(
                filters[1], filters[2], kernel_size=3, stride=2, padding=1
            )
        else:
            self.conv3 = nn.Conv2d(
                filters[1], filters[2], kernel_size=3, stride=2, padding=0
            )

        # Calculate the flattened size after conv layers
        # This needs to be computed based on the actual input dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            self.hidden_conv_shape = x.shape[1:]
            self.flattened_size = x.numel()

        # Dense layers for embedding
        self.embed_dense = nn.Linear(self.flattened_size, hidden_units)

        # Decoder layers
        self.decode_dense = nn.Linear(hidden_units, self.flattened_size)

        if self.pad3 == "same":
            self.deconv1 = nn.ConvTranspose2d(
                filters[2],
                filters[1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        else:
            self.deconv1 = nn.ConvTranspose2d(
                filters[2],
                filters[1],
                kernel_size=3,
                stride=2,
                padding=0,
                output_padding=1,
            )

        self.deconv2 = nn.ConvTranspose2d(
            filters[1], filters[0], kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            filters[0],
            input_shape[2],
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten and embed
        x_flat = x.view(x.size(0), -1)
        h = self.embed_dense(x_flat)

        # L2 normalization (equivalent to Lambda layer in TF)
        h_norm = F.normalize(h, p=2, dim=-1)

        # Decoder
        x_decode = F.relu(self.decode_dense(h_norm))
        x_decode = x_decode.view(x_decode.size(0), *self.hidden_conv_shape)

        x_decode = F.relu(self.deconv1(x_decode))
        x_decode = F.relu(self.deconv2(x_decode))
        x_decode = self.deconv3(x_decode)  # No activation on final layer

        # Flatten reconstructed output
        x_decode_flat = x_decode.view(x_decode.size(0), -1)

        # Concatenate embedding and reconstruction (equivalent to Concatenate layer)
        output = torch.cat([h_norm, x_decode_flat], dim=1)

        return output, h_norm, x_decode


def model_conv(cfg, load_weights=True):
    """
    Create the convolutional autoencoder model

    Args:
        cfg: Configuration object with INPUT_SHAPE, CLUSTER.HIDDEN_UNITS, AUTOENCODER.WEIGTH_PATH
        load_weights: Whether to load pre-trained weights

    Returns:
        model: PyTorch model
    """
    input_shape = cfg.INPUT_SHAPE  # Assuming (H, W, C) format
    hidden_units = cfg.CLUSTER.HIDDEN_UNITS
    weight_path = cfg.AUTOENCODER.WEIGTH_PATH

    model = ConvAutoencoder(input_shape, hidden_units)

    if load_weights and os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        print(f"model_conv: weights was loaded, weight path is {weight_path}")

    return model


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-8, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def loss_train_base(y_true, y_pred, hidden_units):
    """
    Custom loss function equivalent to the TensorFlow version

    Args:
        y_true: Ground truth (original input flattened)
        y_pred: Model output (concatenated embedding + reconstruction)
        hidden_units: Size of embedding dimension

    Returns:
        loss: MSE between true input and reconstructed input
    """
    # Extract only the reconstruction part (skip embedding part)
    y_pred_recon = y_pred[:, hidden_units:]

    # Flatten ground truth if needed
    if len(y_true.shape) > 2:
        y_true = y_true.view(y_true.size(0), -1)

    return F.mse_loss(y_true, y_pred_recon)


def train_base(
    model,
    dataloader: DataLoader,
    cfg,
    epoch=None,
    device: torch.device = torch.device("cpu"),
):
    """
    Training function for the base autoencoder

    Args:
        model: PyTorch model
        dataloader: DataLoader with training data
        cfg: Configuration object
        epoch: Number of epochs (optional, uses cfg if not provided)
        device: Device to train on ('cpu' or 'cuda')
    """
    if epoch is None:
        epoch = cfg.AUTOENCODER.AUTOENCODER_EPOCHS

    hidden_units = cfg.CLUSTER.HIDDEN_UNITS
    weight_path = cfg.AUTOENCODER.WEIGTH_PATH

    model.to(device)
    optimizer = optim.Adam(model.parameters())
    early_stopping = EarlyStopping(patience=5, min_delta=1e-8)

    pbar = tqdm(total=epoch, desc="Training")

    for ep in range(epoch):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)):
                # If dataloader returns (data, target), use data
                batch_data = batch_data[0]

            batch_data = batch_data.to(device)

            # Forward pass
            output, embedding, reconstruction = model(batch_data)

            # Calculate loss
            loss = loss_train_base(batch_data, output, hidden_units)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Update progress bar
        pbar.set_postfix({"loss": avg_loss})
        pbar.update(1)

        # Early stopping
        if early_stopping(avg_loss, model):
            print(f"Early stopping at epoch {ep + 1}")
            break

    pbar.close()

    # Save model weights
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    torch.save(model.state_dict(), weight_path)
    print(f"Model weights saved to {weight_path}")
