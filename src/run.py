import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.LoadData import load_dataset
from trainers.SelfAdjustGraphTrainer import SelfAdjustGraphTrainer
from trainers.SpectralNetTrainer import SpectralNetTrainer
from utils.Config import Config
from utils.Metrics import run_evaluate_with_labels

# Configuration for SelfAdjustGraphTrainer
config_dict = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "training": {"lr": 0.0005, "num_epoch": 100},
    "self_adjust_graph": {
        "g_dim": 128,
        "gamma": 1,
        "mu": 1,
        "delta": 0.1,
        "cluster": 10,  # Number of clusters
    },
    "school": {
        "k": 10,
        "out_feat": 256,
        "gcn_hid_units": 512,
        "gcn_out_size": 256,
        "spectral_architecture": [256, 256, 128],
    },
    "dataset": {"dataset": "MSRC-v2", "batch_size": 512},
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

    features = torch.load("dataset\\resnet\\MSRC-v2_Feature.pt")
    labels = torch.load("dataset\\resnet\\MSRC-v2_Label.pt")

    n_cluster = len(torch.unique(labels))

    config.self_adjust_graph.cluster = n_cluster
    config.school.spectral_architecture.append(n_cluster)
    config.school.feat_size = features.shape[1]
    val_results = []
    losses = []
    for i in range(10):
        # Create trainer
        trainer = SelfAdjustGraphTrainer(config)
        # Train the model
        loss = trainer.train(features=features, labels=labels)
        cluster_assignment = trainer.predict(
            X=features, n_clusters=n_cluster, use_weight=True
        )
        y_target = labels.detach().cpu().numpy()
        result = run_evaluate_with_labels(
            cluster_assignments=cluster_assignment, y=y_target, n_clusters=n_cluster
        )
        losses.append(loss)
        val_results.append(result)

    loss_df = pd.DataFrame(losses)
    loss_df.to_csv(
        f"output\\custom_spectralnet_kmeans\\{config.dataset.dataset}_loss.csv"
    )
    val_df = pd.DataFrame(val_results)
    val_df.to_csv(
        f"output\\custom_spectralnet_kmeans\\{config.dataset.dataset}_val.csv"
    )


def run_spectral_net():
    # Create config object
    config = Config.from_dict(config_dict)

    # Load MSRC-v2 dataset and extract features

    features = torch.load("dataset\\resnet\\MSRC-v2_Feature.pt")
    labels = torch.load("dataset\\resnet\\MSRC-v2_Label.pt")

    n_cluster = len(torch.unique(labels))

    config.self_adjust_graph.cluster = n_cluster
    config.spectral.architecture.append(n_cluster)
    config.siamese.architecture.append(n_cluster)

    val_results = []
    losses = []
    for i in range(10):
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

    loss_df = pd.DataFrame(losses)
    loss_df.to_csv(f"output\\spectralnet\\{config.dataset.dataset}_loss.csv")
    val_df = pd.DataFrame(val_results)
    val_df.to_csv(f"output\\spectralnet\\{config.dataset.dataset}_val.csv")


if __name__ == "__main__":
    run_self_adjust_graph_net()
