import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.trainers.AutoEncoderTrainer import AutoEncoderTrainer
from utils.Config import AutoEncoderConfig


def load_mnist():
    # Create the MNIST directory if it doesn't exist
    mnist_dir = "dataset"
    os.makedirs(mnist_dir, exist_ok=True)
    transform_data = transforms.Compose([transforms.ToTensor()])
    # Download MNIST training set
    train_set = datasets.MNIST(
        root=mnist_dir, train=True, download=True, transform=transform_data
    )

    # Download MNIST test set
    test_set = datasets.MNIST(
        root=mnist_dir, train=False, download=True, transform=transform_data
    )

    print("MNIST dataset has been downloaded successfully!")
    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")

    x_train, y_train = zip(*train_set)
    x_train, y_train = torch.cat(x_train), torch.Tensor(y_train)
    x_test, y_test = zip(*test_set)
    x_test, y_test = torch.cat(x_test), torch.Tensor(y_test)

    return x_train, y_train, x_test, y_test


def load_usps_mnist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usps_mnist_dir = "dataset/USPS.h5"
    with h5py.File(usps_mnist_dir, "r") as f:
        x = np.array(f["x"][:])
        y = np.array(f["y"][:])

    features = torch.from_numpy(x).to(device)
    labels = torch.from_numpy(y).to(device)

    dataset = torch.utils.data.TensorDataset(features, labels)

    return dataset


def create_mnist_feature(is_usps_mnist: bool = False):
    ae_config = AutoEncoderConfig()

    if not is_usps_mnist:
        x_train, y_train, x_test, y_test = load_mnist()

        X_origin = torch.cat([x_train, x_test])

        if y_train is not None:
            y_origin = torch.cat([y_train, y_test])
        else:
            y_origin = None
    else:
        dataset = load_usps_mnist()
        X_origin = dataset.tensors[0]
        y_origin = dataset.tensors[1]

    trainset_len = int(len(X_origin) * 0.9)
    validset_len = len(X_origin) - trainset_len
    trainset, _ = random_split(X_origin, [trainset_len, validset_len])
    train_loader = DataLoader(trainset, batch_size=ae_config.batch_size, shuffle=True)
    # valid_loader = DataLoader(validset, batch_size=ae_config.batch_size, shuffle=False)
    # n_clusters = len(torch.unique(y_origin))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae_trainer = AutoEncoderTrainer(config=ae_config, device=device)

    ae_trainer.train(train_loader=train_loader, valid_loader=None)

    X = ae_trainer.embed(X_origin)

    if is_usps_mnist:
        torch.save(X, "dataset/embedding/auto_encoder/USPS_Feature.pt")
        torch.save(y_origin, "dataset/embedding/auto_encoder/USPS_Label.pt")
    else:
        torch.save(X, "dataset/embedding/auto_encoder/mnist_raw_Feature.pt")
        torch.save(y_origin, "dataset/embedding/auto_encoder/mnist_raw_Label.pt")


if __name__ == "__main__":
    create_mnist_feature(is_usps_mnist=True)
