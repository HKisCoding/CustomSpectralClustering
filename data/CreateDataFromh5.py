import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.trainers.AutoEncoderTrainer import AutoEncoderTrainer
from utils.Config import AutoEncoderConfig


def create_feature_from_h5(dataset_name: str, h5_file_path: str):
    ae_config = AutoEncoderConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(h5_file_path, "r") as f:
        x = np.array(f["x"][:])
        y = np.array(f["y"][:])

    features = torch.from_numpy(x).float().to(device)
    labels = torch.from_numpy(y).float().to(device)

    dataset = torch.utils.data.TensorDataset(features, labels)

    X_origin = dataset.tensors[0]
    y_origin = dataset.tensors[1]

    trainset_len = int(len(X_origin) * 1.0)
    validset_len = len(X_origin) - trainset_len
    trainset, _ = random_split(X_origin, [trainset_len, validset_len])
    train_loader = DataLoader(trainset, batch_size=ae_config.batch_size, shuffle=True)
    # valid_loader = DataLoader(validset, batch_size=ae_config.batch_size, shuffle=False)
    # n_clusters = len(torch.unique(y_origin))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae_trainer = AutoEncoderTrainer(config=ae_config, device=device)

    ae_trainer.train(train_loader=train_loader, valid_loader=None)

    X = ae_trainer.embed(X_origin)

    torch.save(X, f"dataset/embedding/auto_encoder/{dataset_name}_Feature.pt")
    torch.save(y_origin, f"dataset/embedding/auto_encoder/{dataset_name}_Label.pt")


if __name__ == "__main__":
    dataset_name = "fashion_mnist"
    h5_file_path = "dataset/FASHION.h5"
    create_feature_from_h5(dataset_name, h5_file_path)
