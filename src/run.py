import pandas as pd
import torch

from trainers.SelfAdjustGraphTrainer import SelfAdjustGraphTrainer
from trainers.SpectralNetTrainer import SpectralNetTrainer
from utils.Config import Config
from utils.Metrics import run_evaluate_with_labels

# Configuration for SelfAdjustGraphTrainer
config_dict = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "training": {"lr": 0.0005, "num_epoch": 100},
    "self_adjust_graph": {
        "g_dim": 256,
        "gamma": 1,
        "mu": 1,
        "delta": 0.1,
        "cluster": 10,  # Number of clusters
        "auxillary_loss_kind": "entropy",
        "auxillary_loss_alpha": 0.1,
    },
    "school": {
        "k": 10,
        "feat_size": 512,
        "out_feat": 256,
        "gcn_architecture": [512, 256, 256],
        "spectral_architecture": [1024, 1024, 256],
    },
    "dataset": {"dataset": "Caltech_101", "batch_size": 1024},
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

    features = +torch.load(config.dataset.data_path[dataset]["features"])
    labels = torch.load(config.dataset.data_path[dataset]["labels"])

    n_cluster = len(torch.unique(labels))

    config.self_adjust_graph.cluster = n_cluster
    config.school.spectral_architecture.append(n_cluster)
    config.school.feat_size = features.shape[1]
    val_results = []
    losses = []
    for i in range(1):
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
            # losses.append(loss)
            val_results.append(result)

        loss_df = pd.DataFrame(loss)
        loss_df.to_csv(
            f"output\\self_adjust_graph_with_soft_assignment\\{config.dataset.dataset}\\train_time{i + 1}_{config.training.num_epoch}epochs_loss.csv"
        )
        val_df = pd.DataFrame(val_results)
        val_df.to_csv(
            f"output\\self_adjust_graph_with_soft_assignment\\{config.dataset.dataset}\\train_time{i + 1}_{config.training.num_epoch}epochs_val.csv"
        )


def run_spectral_net():
    # Create config object
    config = Config.from_dict(config_dict)

    # Load MSRC-v2 dataset and extract features

    dataset = config.dataset.dataset

    features = torch.load(config.dataset.data_path[dataset]["features"])
    labels = torch.load(config.dataset.data_path[dataset]["labels"])

    n_cluster = len(torch.unique(labels))

    config.self_adjust_graph.cluster = n_cluster
    config.spectral.architecture.append(n_cluster)
    config.siamese.architecture.append(n_cluster)

    val_results = []
    losses = []
    for i in range(1):
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
        loss_df.to_csv(
            f"output\\spectralnet\\{config.dataset.dataset}\\train_time{i + 1}_{config.training.num_epoch}epochs_loss.csv"
        )
        val_df = pd.DataFrame(val_results)
        val_df.to_csv(
            f"output\\spectralnet\\{config.dataset.dataset}\\train_time{i + 1}_{config.training.num_epoch}epochs_val.csv"
        )


def run_validation():
    # Create config object
    config = Config.from_dict(config_dict)

    # Load MSRC-v2 dataset and extract features

    dataset = config.dataset.dataset

    features = torch.load(config.dataset.data_path[dataset]["features"])
    labels = torch.load(config.dataset.data_path[dataset]["labels"])

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
            use_weight="weights\\self_adjust_graph\\best_spectral_loss.pt",
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


if __name__ == "__main__":
    # run_spectral_net()
    # run_self_adjust_graph_net()
    run_validation()
