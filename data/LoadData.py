import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from src.models.backbones.resnet import ResNet
from utils.Config import Config


class ImageDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annotation: str,
        config: Config | None = None,
        use_extractor: bool = False,
    ):
        self.image_dir = image_dir
        self.annotation = pd.read_csv(annotation, index_col=0)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.PILToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Feature extraction setup
        self.config = config
        self.use_extractor = use_extractor
        if use_extractor and config is not None:
            self.model = ResNet(config.backbone.name, retrain=False).get_extractor()
            self.model = self.model.to(self.device)
            self.model.eval()

    def _extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features from a single image using the backbone model.

        Args:
            image: Input image tensor

        Returns:
            Extracted features tensor
        """
        if self.config is None:
            raise ValueError("Config must be provided when use_extractor is True")

        with torch.no_grad():
            feat = self.model(image)
            feat = feat.squeeze()
        return feat

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        img_name = self.annotation.iloc[index, 0]
        label_name = self.annotation.iloc[index, 1]
        label = self.annotation.iloc[index, 2]

        img_sub_dir = os.path.join(label_name, img_name)
        img_path = os.path.join(self.image_dir, img_sub_dir)
        img = Image.open(img_path).convert("RGB")

        image_tensor = self.transform(img).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)

        if self.use_extractor:
            features = self._extract_features(image_tensor)
            return features, label
        else:
            return image_tensor, label


def load_dataset(
    dataset_name: str, config: Config, val_size: float = 0.2, use_extractor: bool = True
) -> tuple[Dataset, Dataset]:
    """Load dataset and optionally extract features.

    Args:
        dataset_name: Name of the dataset to load
        config: Configuration object
        val_size: Validation set size ratio
        use_extractor: Whether to extract features using the backbone model

    Returns:
        tuple: (trainset, validset)
        If use_extractor is True:
            trainset and validset will contain pre-extracted features and labels
        If use_extractor is False:
            trainset and validset will contain transformed images and labels
    """
    if dataset_name == "Caltech_101":
        img_dir = "dataset/Caltech_101"
        annotation = "dataset/Caltech_101.csv"

        # Create full dataset
        dataset = ImageDataset(
            image_dir=img_dir,
            annotation=annotation,
            config=config if use_extractor else None,
            use_extractor=use_extractor,
        )

        # Split into train and validation sets
        trainset_len = int(len(dataset) * 0.7)
        validset_len = len(dataset) - trainset_len
        trainset, validset = random_split(dataset, [trainset_len, validset_len])

        return trainset, validset
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported")
