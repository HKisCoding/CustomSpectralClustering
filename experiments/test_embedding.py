import numpy as np
import torch

from src.trainers.EmbeddingTrainer import ManifoldsEmbeddingTrainer
from utils.Config import (
    Config,
    DatasetConfig,
    ModelConfig,
    OptimizationConfig,
    SelfAdjustGraphConfig,
    TrainingConfig,
)


def get_feature_labels(feature_path, labels_path):
    X = torch.load(feature_path)
    y = torch.load(labels_path)

    return X, y


feature_path = "dataset/MRSC_Feature.pt"
labels_path = "dataset/MRSC_label.pt"
X, y = get_feature_labels(feature_path, labels_path)

n_clusters = len(torch.unique(y))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create config groups
dataset_config = DatasetConfig(dataset="Caltech_101", batch_size=32)

training_config = TrainingConfig(
    lr=0.001, num_epoch=50, weight_path="output/weights/Caltech_101", save_every=10
)

model_config = ModelConfig(backbone="resnet18", feature_dims=512, z_dims=64)

optimization_config = OptimizationConfig(eps=0.1, momentum=0.9, weight_decay=5e-4)

self_adjust_graph_config = SelfAdjustGraphConfig()

# Create main config
config = Config(
    dataset=dataset_config,
    training=training_config,
    model=model_config,
    optimization=optimization_config,
    self_adjust_graph=self_adjust_graph_config,
)

embedding = ManifoldsEmbeddingTrainer(config, device)

embedding.train(X=X, y=y)
