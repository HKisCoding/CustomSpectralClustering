import torch
from torch.utils.data import DataLoader

from data.LoadData import load_dataset
from trainers.SelfAdjustGraphTrainer import SelfAdjustGraphTrainer
from utils.Config import Config

# Configuration for SelfAdjustGraphTrainer
config_dict = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "training": {
        "lr": 0.002,
        "num_epoch": 100,
        "batch_size": 32,
    },
    "self_adjust_graph": {
        "hid_unit": 256,
        "feat_size": 512,
        "out_feat": 256,
        "g_dim": 128,
        "k": 10,
        "gamma": 0.1,
        "mu": 0.1,
        "delta": 0.1,
        "cluster": 10,  # Number of clusters
    },
    "dataset": {"dataset": "Caltech_101", "batch_size": 32},
    "backbone": {
        "name": "resnet18",
        "pretrained": True,
        "feature_dims": 512,
        "z_dims": 256,
    },
}


def main():
    # Create config object
    config = Config.from_dict(config_dict)

    # Create trainer
    trainer = SelfAdjustGraphTrainer(config)

    # Load Caltech_101 dataset and extract features
    trainset, validset = load_dataset("Caltech_101", config)

    # Extract features from datasets
    train_loader = DataLoader(
        trainset, batch_size=config.dataset.batch_size, shuffle=False
    )
    valid_loader = DataLoader(
        validset, batch_size=config.dataset.batch_size, shuffle=False
    )

    # Collect all features and labels
    train_features = []
    train_labels = []
    valid_features = []
    valid_labels = []

    with torch.no_grad():
        for features, labels in train_loader:
            train_features.append(features)
            train_labels.append(labels)
        for features, labels in valid_loader:
            valid_features.append(features)
            valid_labels.append(labels)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    valid_features = torch.cat(valid_features, dim=0)
    valid_labels = torch.cat(valid_labels, dim=0)

    # Combine training and validation features
    features = torch.cat([train_features, valid_features], dim=0)
    labels = torch.cat([train_labels, valid_labels], dim=0)

    # Train the model
    trainer.train(features=features, labels=labels)

    # You can add evaluation code here if needed


if __name__ == "__main__":
    main()
