import glob
import os
import re

import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
from torch.nn import Sequential
from torch.utils.data import DataLoader, Dataset, random_split

from data.LoadMatData import read_mat_data
from src.models.backbones.resnet import ResNet
from utils.Config import Config


class ImageDataset(Dataset):
    def __init__(
        self, image_dir: str, annotation: str, device: torch.device, add_dim=False
    ):
        self.image_dir = image_dir
        self.annotation = pd.read_csv(
            annotation, index_col=0, converters={"label_name": str}
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
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
    dataset_name: str,
    train_size: float = 0.8,
    split_dataset: bool = True,
    device: torch.device = torch.device("cpu"),
) -> tuple[Dataset, Dataset] | Dataset:
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
        img_dir = f"dataset/{dataset_name}"
        annotation = f"dataset/{dataset_name}.csv"

        # Create full dataset
        dataset = ImageDataset(image_dir=img_dir, annotation=annotation, device=device)

        # Split into train and validation sets
        if split_dataset:
            trainset_len = int(len(dataset) * train_size)
            validset_len = len(dataset) - trainset_len
            trainset, validset = random_split(dataset, [trainset_len, validset_len])
            return trainset, validset
        else:
            return dataset
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

    torch.save(features, f"dataset/embedding/resnet/{dataset_name}_Feature.pt")
    torch.save(labels, f"dataset/embedding/resnet/{dataset_name}_Label.pt")


def create_features_object(dataset_name, backbone_name, device):
    model = ResNet(backbone_name).get_extractor()
    model.eval()
    model.to(torch.device(device))

    load_feature_from_scratch(dataset_name, model, device=device)


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


def create_usps_mnist_features(
    backbone_name: str = "resnet18", device: str = "cpu", batch_size: int = 32
):
    """
    Load USPS MNIST dataset using torchvision ImageFolder and create feature.pt and label.pt files from backbone model.

    Args:
        backbone_name: Name of the backbone model to use for feature extraction
        device: Device to run the model on ('cpu' or 'cuda')
        batch_size: Batch size for processing
    """
    try:
        # Define transforms matching ResNet expectations
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                transforms.Resize((32, 32)),
                transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1,1]
            ]
        )

        # Load the dataset using ImageFolder
        dataset_path = "dataset/MNIST/Numerals"
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Initialize the backbone model
        model = ResNet(backbone_name).get_extractor()
        model.eval()
        model.to(torch.device(device))

        features = []
        labels = []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                feats = model(imgs)
                features.append(feats.cpu())
                labels.append(lbls.cpu())

        # Concatenate all batches
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        # Create output directory
        os.makedirs("dataset/resnet", exist_ok=True)

        # Save features and labels
        torch.save(features, "dataset/resnet/usps_mnist_Feature.pt")
        torch.save(labels, "dataset/resnet/usps_mnist_Label.pt")

        print(f"Saved features with shape: {features.shape}")
        print(f"Saved labels with shape: {labels.shape}")
        print("USPS MNIST features and labels saved successfully!")

    except Exception as e:
        print(f"Error loading USPS MNIST dataset: {e}")
        raise


def create_mnist_ae_features(device: str = "cpu", batch_size: int = 512):
    from src.trainers.AutoEncoderTrainer import AutoEncoderTrainer
    from utils.Config import Config

    # Define transforms for USPS MNIST data - simple grayscale conversion and normalization
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
        ]
    )

    # Load the dataset using ImageFolder
    dataset_path = "dataset/MNIST/Numerals"
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the backbone model
    trainer = AutoEncoderTrainer(
        config=Config().auto_encoder,
        device=torch.device(device),
    )

    trainer.train(loader, None)

    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            feats = trainer.embed(imgs)
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


def create_mnist_features(
    batch_size: int = 512,
):  # Define transforms for USPS MNIST data - simple grayscale conversion and normalization
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
        ]
    )

    dataset_path = "dataset/MNIST/Numerals"
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    features = []
    labels = []
    for imgs, lbls in loader:
        features.append(imgs)
        labels.append(lbls)

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


def create_mat_features(
    dataset_name, mat_file_path, output_dir="dataset/embedding/mat_file"
):
    """
    Extract first view from train and test datasets and save as .pt files

    Args:
        mat_file_path: Path to the .mat file
        output_dir: Directory to save the output files
    """
    print(f"Loading data from: {mat_file_path}")

    # Load the datasets
    read_mat_data(
        str_name=mat_file_path,
        dataset_name=dataset_name,
        output_dir=output_dir,
        save_feature_label=True,
    )


def create_caltech_101_annotation_csv():
    img_dir = "dataset\\Caltech_101"
    output_csv = "dataset\\Caltech_101.csv"
    data = []

    # Get all category directories
    categories = sorted(
        [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    )

    # Create label mapping: category_name -> numeric label
    label_map = {cat: idx for idx, cat in enumerate(categories)}

    idx = 0
    for label_name in categories:
        category_dir = os.path.join(img_dir, label_name)
        # Get all jpg files in this category
        files = glob.glob(os.path.join(category_dir, "*.jpg"))
        for f in sorted(files):
            base = os.path.basename(f)
            label = label_map[label_name]
            data.append([idx, base, label_name, label])
            idx += 1

    df = pd.DataFrame(data, columns=["", "img_name", "label_name", "label"])
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    backbone_name = Config().backbone.name

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create_caltech_101_annotation_csv()

    # create_mat_features(
    #     dataset_name="animal-50",
    #     mat_file_path="dataset/animal.mat",
    #     output_dir="dataset/embedding/mat_file",
    # )

    # create_coil20_annotation_csv()

    create_features_object(
        dataset_name="Caltech_101", backbone_name=backbone_name, device=device
    )
    # create_mnist_features_object("dataset/MNIST")

    # Example: Create USPS MNIST features
    # create_usps_mnist_features(
    #     backbone_name=backbone_name, device=device, batch_size=1024
    # )
    # create_mnist_ae_features(device=device, batch_size=1024)
    # create_mnist_features()
