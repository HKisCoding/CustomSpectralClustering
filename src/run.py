import os

import pandas as pd
import torch

# Import necessary modules for AutoEncoder training
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.trainers.AutoEncoderTrainer import AutoEncoderTrainer
from trainers.AdjustiveGraphEncoderTrainer import SelfAdjustGraphEncoderTrainer
from trainers.SelfAdjustGraphTrainer import SelfAdjustGraphTrainer
from trainers.SpectralNetTrainer import SpectralNetTrainer
from utils.Config import Config
from utils.Metrics import run_evaluate_with_labels

# Configuration for SelfAdjustGraphTrainer
config_dict = {
    "training": {"lr": 0.0005, "num_epoch": 100},
    "self_adjust_graph": {
        "g_dim": 32,
        "gamma": 1,
        "mu": 0.1,
        "delta": 1,
        "theta": 1,
        "cluster": 10,  # Number of clusters
        "auxillary_loss_kind": "entropy",
        "auxillary_loss_alpha": 0.1,
    },
    "school": {
        "k": 10,
        "feat_size": 512,
        "out_feat": 512,
        "gae_architecture": [1024, 1024, 512],
        "gcn_architecture": [1024, 512],
        "spectral_architecture": [1024, 1024, 512],
    },
    "dataset": {"dataset": "MSRC-v2", "batch_size": 2000},
    "backbone": {
        "name": "resnet18",
        "pretrained": True,
        "feature_dims": 512,
        "z_dims": 256,
    },
    "spectral": {"architecture": [1024, 1024, 512], "scale_k": 10},
}


def run_self_adjust_graph_net():
    # Create config object
    config = Config.from_dict(config_dict)

    # Load MSRC-v2 dataset and extract features

    dataset = config.dataset.dataset

    features = torch.load(config.dataset.data_path[dataset]["features"])
    features = features.float()
    labels = torch.load(config.dataset.data_path[dataset]["labels"]).squeeze()
    labels = labels.float()

    if config.dataset.batch_size > features.shape[0]:
        config.dataset.batch_size = features.shape[0]

    n_cluster = len(torch.unique(labels))

    config.self_adjust_graph.cluster = n_cluster
    config.school.spectral_architecture.append(n_cluster)
    config.school.feat_size = features.shape[1]
    val_results = []
    losses = []
    output_path = (
        f"output\\self_adjust_graph_with_soft_assignment\\{config.dataset.dataset}"
    )
    os.makedirs(output_path, exist_ok=True)
    for i in range(5):
        # Create trainer
        loss = 0
        trainer = SelfAdjustGraphTrainer(config)
        try:
            # Train the model
            loss = trainer.train(features=features, labels=labels)
        except Exception as e:
            print(e)
            continue
        finally:
            cluster_assignment = trainer.predict(
                X=features, n_clusters=n_cluster, use_weight=None
            )
            y_target = labels.detach().cpu().numpy()
            result = run_evaluate_with_labels(
                cluster_assignments=cluster_assignment, y=y_target, n_clusters=n_cluster
            )
            losses.append(loss)
            val_results.append(result)

            # loss_df = pd.DataFrame(losses)
            # loss_df.to_csv(
            #     f"{output_path}\\traintime{i + 1}_{config.training.num_epoch}epochs_loss.csv"
            # )
            val_df = pd.DataFrame(val_results)
            val_df.to_csv(f"{output_path}\\{config.training.num_epoch}epochs_val.csv")

        # os.makedirs(
        #     os.path.join(
        #         trainer.weight_path,
        #         config.dataset.dataset,
        #     ),
        #     exist_ok=True,
        # )
        # torch.save(
        #     {
        #         "model_state_dict": trainer.model.state_dict(),
        #         "optimizer_state_dict": trainer.optimizer.state_dict(),
        #         "orthonorm_weights": trainer.model.spectral_net.orthonorm_weights,
        #     },
        #     os.path.join(
        #         trainer.weight_path,
        #         config.dataset.dataset,
        #         "self_adjust_graph_with_soft_assignment.pt",
        #     ),
        # )


def run_spectral_net():
    # Create config object
    config = Config.from_dict(config_dict)

    # Load MSRC-v2 dataset and extract features

    dataset = config.dataset.dataset

    features = torch.load(config.dataset.data_path[dataset]["features"])
    features = features.float()
    labels = torch.load(config.dataset.data_path[dataset]["labels"]).squeeze()
    labels = labels.float()

    n_cluster = len(torch.unique(labels))

    config.self_adjust_graph.cluster = n_cluster
    config.spectral.architecture.append(n_cluster)
    config.siamese.architecture.append(n_cluster)

    val_results = []
    losses = []
    for i in range(5):
        # Create trainer
        trainer = SpectralNetTrainer(config, device=config.device, is_sparse=False)
        # Train the model
        _, loss = trainer.train(X=features, y=labels)

        cluster_assignment = trainer.predict(X=features, n_clusters=n_cluster)
        y_target = labels.detach().cpu().numpy()
        result = run_evaluate_with_labels(
            cluster_assignments=cluster_assignment, y=y_target, n_clusters=n_cluster
        )
        losses.append(loss)
        val_results.append(result)

        output_path = f"output\\spectralnet\\{config.dataset.dataset}"
        os.makedirs(output_path, exist_ok=True)

        loss_df = pd.DataFrame(losses)
        loss_df.to_csv(
            f"{output_path}\\traintime{i + 1}_{config.training.num_epoch}epochs_loss.csv"
        )
        val_df = pd.DataFrame(val_results)
        val_df.to_csv(
            f"{output_path}\\traintime{i + 1}_{config.training.num_epoch}epochs_val.csv"
        )


def run_validation():
    # Create config object
    config = Config.from_dict(config_dict)

    # Load MSRC-v2 dataset and extract features

    dataset = config.dataset.dataset

    features = torch.load(config.dataset.data_path[dataset]["features"])
    labels = torch.load(config.dataset.data_path[dataset]["labels"]).squeeze()

    n_cluster = len(torch.unique(labels))

    config.self_adjust_graph.cluster = n_cluster
    config.school.spectral_architecture.append(n_cluster)
    config.school.feat_size = features.shape[1]

    val_results = []
    for i in range(5):
        trainer = SelfAdjustGraphTrainer(config)
        cluster_assignment = trainer.predict(
            X=features,
            n_clusters=n_cluster,
            use_weight=f"{config.dataset.dataset}/self_adjust_graph_with_soft_assignment.pt",
        )
        y_target = labels.detach().cpu().numpy()
        result = run_evaluate_with_labels(
            cluster_assignments=cluster_assignment, y=y_target, n_clusters=n_cluster
        )
        val_results.append(result)

    # val_df = pd.DataFrame(val_results)
    # val_df.to_csv(
    #     f"output\\self_adjust_graph_with_soft_assignment\\{config.dataset.dataset}_val.csv"
    # )


def run_training_auto_encoder():
    config = Config.from_dict(config_dict)
    autoencoder_config = config.auto_encoder

    device = config.device
    print(f"Using device: {device}")

    # Define transforms for USPS MNIST data - simple grayscale conversion and normalization
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Resize((32, 32)),
            transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1,1]
        ]
    )

    # Load USPS MNIST dataset from dataset/MNIST/Numerals
    dataset_path = "dataset/MNIST/Numerals"
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Create data loader to load all data at once
    data_loader = DataLoader(
        dataset,
        batch_size=autoencoder_config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    # Create output directory for weights
    os.makedirs(os.path.dirname(autoencoder_config.weight_path), exist_ok=True)

    # Create and train autoencoder
    ae_trainer = AutoEncoderTrainer(autoencoder_config, device)
    print("Starting AutoEncoder training...")
    ae_trainer.train(data_loader, None)

    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in data_loader:
            imgs = imgs.to(device)
            feats = ae_trainer.embed(imgs)
            features.append(feats.cpu())
            labels.append(lbls.cpu())

    # Concatenate all batches
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    # Create output directory
    os.makedirs("dataset/embedding/auto_encoder", exist_ok=True)

    # Save features and labels
    torch.save(features, "dataset/embedding/auto_encoder/usps_mnist_Feature.pt")
    torch.save(labels, "dataset/embedding/auto_encoder/usps_mnist_Label.pt")

    print(f"Saved features with shape: {features.shape}")
    print(f"Saved labels with shape: {labels.shape}")
    print("USPS MNIST features and labels saved successfully!")

    print("AutoEncoder training completed!")


if __name__ == "__main__":
    # run_spectral_net()
    run_self_adjust_graph_net()
    # run_validation()
    # run_training_auto_encoder()
