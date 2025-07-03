import glob
import os
import re

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn import Sequential
from torch.utils.data import DataLoader, Dataset, random_split

from src.models.backbones.resnet import ResNet
from utils.Config import Config


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, annotation: str, device="cpu", add_dim=False):
        self.image_dir = image_dir
        self.annotation = pd.read_csv(
            annotation, index_col=0, converters={"label_name": str}
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.device = torch.device(device)
        self.add_dim = add_dim

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        img_name = self.annotation.iloc[index, 0]
        label_name = self.annotation.iloc[index, 1]
        label = self.annotation.iloc[index, 2]

        img_path = os.path.join(self.image_dir, label_name, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        image_tensor = self.transform(img).to(self.device)
        if self.add_dim:
            image_tensor = image_tensor.unsqueeze(0)

        return image_tensor, label


def load_dataset(
    dataset_name: str, train_size: float = 0.8, device: str = "cpu"
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
    try:
        img_dir = f"dataset\\{dataset_name}"
        annotation = f"dataset\\{dataset_name}.csv"

        # Create full dataset
        dataset = ImageDataset(image_dir=img_dir, annotation=annotation, device=device)

        # Split into train and validation sets
        trainset_len = int(len(dataset) * train_size)
        validset_len = len(dataset) - trainset_len
        trainset, validset = random_split(dataset, [trainset_len, validset_len])

        return trainset, validset
    except:
        raise ValueError(f"Dataset '{dataset_name}' is not supported")


def load_feature_from_scratch(dataset_name: str, model: Sequential, device: str):
    trainset, validset = load_dataset(dataset_name, device=device)

    # Extract features from datasets
    train_loader = DataLoader(trainset, batch_size=32, shuffle=False)
    valid_loader = DataLoader(validset, batch_size=32, shuffle=False)

    # Collect all features and labels
    train_features = []
    train_labels = []
    valid_features = []
    valid_labels = []

    with torch.no_grad():
        for imgs, labels in train_loader:
            features = model(imgs)
            train_features.append(features.squeeze())
            train_labels.append(labels)
        for imgs, labels in valid_loader:
            features = model(imgs)
            valid_features.append(features.squeeze())
            valid_labels.append(labels)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    valid_features = torch.cat(valid_features, dim=0)
    valid_labels = torch.cat(valid_labels, dim=0)

    # Combine training and validation features
    features = torch.cat([train_features, valid_features], dim=0)
    labels = torch.cat([train_labels, valid_labels], dim=0)

    torch.save(features, f"dataset\\resnet\\{dataset_name}_Feature.pt")
    torch.save(labels, f"dataset\\resnet\\{dataset_name}_Label.pt")


def create_features_object(dataset_name, backbone_name, device):
    model = ResNet(backbone_name).get_extractor()
    model.eval()
    model.to(torch.device(device))

    load_feature_from_scratch(dataset_name, model, device=device)


def create_mnist_features_object(data_path):
    # Load MNIST dataset from existing path
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    mnist = datasets.MNIST(
        root=data_path, train=True, download=False, transform=transform
    )
    data_loader = DataLoader(mnist, batch_size=128, shuffle=False)

    # Initialize lists to store features and labels
    features = []
    labels = []

    # Process each batch
    with torch.no_grad():
        for images, batch_labels in data_loader:
            # Flatten the images (28x28 = 784 dimensions)
            batch_features = images.view(images.size(0), -1)
            features.append(batch_features)
            labels.append(batch_labels)

    # Concatenate all batches
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    # Save features and labels
    save_dir = os.path.join(data_path, "processed")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(features, os.path.join(save_dir, "mnist_features.pt"))
    torch.save(labels, os.path.join(save_dir, "mnist_labels.pt"))


def get_colon_cancer(path):
    data = pd.read_csv(path, index_col=0)
    data = data.dropna().reset_index(drop=True)
    enc = LabelEncoder()
    label = enc.fit_transform(data["Dukes Stage"].to_numpy())
    data["Gender"] = enc.fit_transform(data["Gender"].to_numpy())
    data["Location"] = enc.fit_transform(data["Location"].to_numpy())
    feature = data.drop(["ID_REF", "Dukes Stage"], axis=1).to_numpy()

    return torch.tensor(feature).float(), torch.tensor(label)


def create_coil20_annotation_csv():
    img_dir = "dataset\\coil-20"
    output_csv = "dataset\\coil-20.csv"
    files = glob.glob(os.path.join(img_dir, "*.png"))
    data = []
    for idx, f in enumerate(sorted(files)):
        base = os.path.basename(f)
        m = re.match(r"(obj\d+)_", base)
        if not m:
            continue
        label_name = m.group(1)
        label = int(label_name.replace("obj", ""))
        data.append([idx, base, label_name, label])
    df = pd.DataFrame(data, columns=["", "img_name", "label_name", "label"])
    df.to_csv(output_csv, index=False)


def create_image_features_coil20(backbone: str, device="cpu"):
    """
    Extract features for COIL20 images using the given backbone name and save them as .pt files.
    """
    if "resnet" in backbone:
        model = ResNet(backbone).get_extractor()
    else:
        raise NotImplementedError(f"Backbone '{backbone}' not supported.")
    annotation = "dataset/coil-20.csv"
    image_dir = "dataset/coil-20"
    dataset = ImageDataset(image_dir=image_dir, annotation=annotation, device=device)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    features = []
    labels = []
    model.eval()
    model.to(torch.device(device))
    with torch.no_grad():
        for imgs, lbls in loader:
            feats = model(imgs)
            features.append(feats.squeeze())
            labels.append(lbls)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    os.makedirs("dataset/coil-20", exist_ok=True)
    torch.save(features, "dataset/resnet/coil-20_Feature.pt")
    torch.save(labels, "dataset/resnet/coil-20_Label.pt")


def get_leumika(path):
    data = pd.read_csv(path, index_col=0)
    label = data["label"].to_numpy()
    feature = data.drop(["label"], axis=1).to_numpy()

    return torch.tensor(feature).float(), torch.tensor(label)


def get_prokaryotic(path):
    data = loadmat(path)
    label = data["truth"][:, 0]
    feature = data["gene_repert"]

    return torch.tensor(feature).float(), torch.Tensor(label)


if __name__ == "__main__":
    dataset_name = "coil-20"
    backbone_name = Config().backbone.name

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create_coil20_annotation_csv()

    create_features_object(
        dataset_name=dataset_name, backbone_name=backbone_name, device=device
    )
    # create_mnist_features_object("dataset/MNIST")
