import os

import torch
from torchvision import datasets


def download_mnist():
    # Create the MNIST directory if it doesn't exist
    mnist_dir = os.path.join("dataset", "MNIST")
    os.makedirs(mnist_dir, exist_ok=True)

    # Download MNIST training set
    train_dataset = datasets.MNIST(
        root=mnist_dir, train=True, download=True, transform=None
    )

    # Download MNIST test set
    test_dataset = datasets.MNIST(
        root=mnist_dir, train=False, download=True, transform=None
    )

    print("MNIST dataset has been downloaded successfully!")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")


if __name__ == "__main__":
    download_mnist()
