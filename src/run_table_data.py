import pandas as pd
import torch

from data.LoadData import get_colon_cancer, get_leumika, get_prokaryotic
from trainers.SelfAdjustGraphTrainer import SelfAdjustGraphTrainer
from utils.Config import Config
from utils.Metrics import run_evaluate_with_labels

config_dict = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "training": {"lr": 0.001, "num_epoch": 100},
    "self_adjust_graph": {
        "g_dim": 128,
        "gamma": 1,
        "mu": 1,
        "delta": 0.1,
        "cluster": 10,  # Number of clusters
        "auxillary_loss_kind": "entropy",
        "auxillary_loss_alpha": 0.1,
    },
    "school": {
        "k": 10,
        "out_feat": 128,
        "gcn_hid_units": 256,
        "gcn_out_size": 128,
        "spectral_architecture": [512, 512, 128],
    },
    "dataset": {"dataset": "prokaryotic", "batch_size": 256},
}


def run_self_adjust_graph_net():
    # Create config object
    config = Config.from_dict(config_dict)

    dataset = config.dataset.dataset

    if dataset == "colon_cancer":
        features, labels = get_colon_cancer(config.dataset.data_path[dataset]["path"])
    elif dataset == "leumika":
        features, labels = get_leumika(config.dataset.data_path[dataset]["path"])
    elif dataset == "prokaryotic":
        features, labels = get_prokaryotic(config.dataset.data_path[dataset]["path"])
    else:
        raise ValueError(f"Dataset {dataset} not found")

    features = features.to(config.device)
    labels = labels.to(config.device)

    n_cluster = len(torch.unique(labels))

    config.self_adjust_graph.cluster = n_cluster
    config.school.spectral_architecture.append(n_cluster)
    config.school.feat_size = features.shape[1]
    val_results = []
    losses = []
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

        loss_df = pd.DataFrame(losses)
        loss_df.to_csv(
            f"output\\self_adjust_graph_with_soft_assignment\\{config.dataset.dataset}_{config.training.num_epoch}epochs_loss.csv"
        )
        val_df = pd.DataFrame(val_results)
        val_df.to_csv(
            f"output\\self_adjust_graph_with_soft_assignment\\{config.dataset.dataset}_{config.training.num_epoch}epochs_val.csv"
        )


def run_validation():
    # Create config object
    config = Config.from_dict(config_dict)

    # Load MSRC-v2 dataset and extract features

    dataset = config.dataset.dataset

    dataset = config.dataset.dataset

    if dataset == "colon_cancer":
        features, labels = get_colon_cancer(config.dataset.data_path[dataset]["path"])
    elif dataset == "leumika":
        features, labels = get_leumika(config.dataset.data_path[dataset]["path"])
    elif dataset == "prokaryotic":
        features, labels = get_prokaryotic(config.dataset.data_path[dataset]["path"])
    else:
        raise ValueError(f"Dataset {dataset} not found")

    features = features.to(config.device)
    labels = labels.to(config.device)

    n_cluster = len(torch.unique(labels))

    config.self_adjust_graph.cluster = n_cluster
    config.school.spectral_architecture.append(n_cluster)
    config.school.feat_size = features.shape[1]

    val_results = []
    for i in range(1):
        trainer = SelfAdjustGraphTrainer(config)
        cluster_assignment = trainer.predict(
            X=features, n_clusters=n_cluster, use_weight="best_spectral_loss.pt"
        )
        y_target = labels.detach().cpu().numpy()
        result = run_evaluate_with_labels(
            cluster_assignments=cluster_assignment, y=y_target, n_clusters=n_cluster
        )
        val_results.append(result)

    val_df = pd.DataFrame(val_results)
    val_df.to_csv(
        f"output\\self_adjust_graph_with_soft_assignment\\{config.dataset.dataset}_val.csv"
    )


if __name__ == "__main__":
    # run_self_adjust_graph_net()
    run_validation()
