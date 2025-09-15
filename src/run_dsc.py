import os

import torch
from torch.utils.data import DataLoader, Dataset

from data.LoadData import load_dataset
from src.models.net.AE_conv import model_conv
from src.trainers.DSCTrainer import DSC_trainer
from utils.Config import AEConvConfig, Config

# Import necessary modules for AutoEncoder training


# Configuration for SelfAdjustGraphTrainer
config_dict = {
    "training": {"lr": 0.001, "num_epoch": 100},
    "dataset": {"dataset": "coil-20", "batch_size": 256},
    "dsc": {
        "hidden_units": 10,
        "batch_size": 256,
        "n_neighbors": 10,
        "scale_k": 10,
        "n_iter": 20,
        "ae_conv": {
            "batch_size": 256,
            "weight_path": "ae_conv/ae_conv.pth",
            "epochs": 2,
        },
    },
}


def train_dsc():
    config = Config.from_dict(config_dict)
    config.dsc.ae_conv = AEConvConfig(**config_dict["dsc"]["ae_conv"])
    trainer = DSC_trainer(config, config.device)
    batch_size = config.dsc.ae_conv.batch_size

    dataset = config.dataset.dataset

    trainset = load_dataset(dataset, device=config.device, split_dataset=False)

    # Extract features from datasets
    dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            features.append(imgs)
            labels.append(lbls)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    trainer.model = model_conv([batch_size, 3, 224, 224], config, load_weights=False)

    n_cluster = len(torch.unique(labels))

    val_results = []
    losses = []
    for i in range(1):
        if not os.path.exists(trainer.ae_weight_path):
            trainer.train_reconstruction(dataloader)
        trainer.load_reconstruction_weights()

        for feat, label in dataloader:
            loss = trainer.train(feat, n_cluster)
        losses.append(loss)

    val_result = trainer.evaluate(features, labels)
    val_results.append(val_result)

    print(losses)
    print(val_results)

    # output_path = f"output\\dsc\\{config.dataset.dataset}"
    # os.makedirs(output_path, exist_ok=True)
    # loss_df = pd.DataFrame(losses)
    # loss_df.to_csv(f"{output_path}\\loss.csv")
    # val_df = pd.DataFrame(val_results)
    # val_df.to_csv(f"{output_path}\\val.csv")


if __name__ == "__main__":
    train_dsc()
