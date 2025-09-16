import math

import numpy as np
import scipy.io as sio
import torch
from numpy.random import shuffle
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    """
    PyTorch Dataset for multi-view data loaded from MATLAB files
    """

    def __init__(self, data, view_number, labels):
        """
        Construct a MultiViewDataset.
        Args:
            data: List of arrays, one for each view
            view_number: Number of views
            labels: Array of labels
        """
        self.data = {}
        self._num_examples = data[0].shape[0]
        self._labels = torch.from_numpy(labels).long()
        self.view_number = view_number

        for v_num in range(view_number):
            self.data[str(v_num)] = torch.from_numpy(data[v_num]).float()

    def __len__(self):
        return self._num_examples

    def __getitem__(self, idx):
        """
        Get item by index
        Returns:
            Dictionary containing all views and label for the sample at idx
        """
        sample = {}
        for v_num in range(self.view_number):
            sample[f"view_{v_num}"] = self.data[str(v_num)][idx]
        sample["label"] = self._labels[idx]
        return sample

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples


def normalize(data):
    """
    Normalize data to zero mean and unit range
    Args:
        data: Input data array
    Returns:
        Normalized data
    """
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)


def read_mat_data(
    str_name,
    ratio=0.8,
    normalize_data=True,
    dataset_name=None,
    output_dir=None,
    save_feature_label=False,
):
    """
    Read data from MATLAB file and split into train/test sets

    Args:
        str_name: Path to .mat file
        ratio: Training set ratio (default: 0.8)
        normalize_data: Whether to normalize the data (default: True)
        dataset_name: Name of the dataset
        output_dir: Directory to save the output files
        save_feature_label: Whether to save the feature and label files

    Returns:
        train_dataset: MultiViewDataset for training
        test_dataset: MultiViewDataset for testing
        view_number: Number of views in the data
    """
    # Load MATLAB file
    data = sio.loadmat(str_name)

    # Get number of views
    view_number = data["X"].shape[1]

    # Split data by views
    X = np.split(data["X"], view_number, axis=1)

    # Initialize lists for train/test data
    X_train = []
    X_test = []
    labels_train = []
    labels_test = []

    # Handle labels (ensure they start from 1)
    if min(data["gt"]) == 0:
        labels = data["gt"] + 1
    else:
        labels = data["gt"]

    classes = max(labels)[0]
    all_length = 0

    # Split data class by class to maintain class balance
    for c_num in range(1, classes + 1):
        # Get indices for current class
        c_length = np.sum(labels == c_num)
        index = np.arange(c_length)
        shuffle(index)

        # Split labels
        train_size = math.floor(c_length * ratio)
        train_indices = index[:train_size]
        test_indices = index[train_size:]

        labels_train.extend(labels[all_length + train_indices])
        labels_test.extend(labels[all_length + test_indices])

        # Split data for each view
        X_train_temp = []
        X_test_temp = []

        for v_num in range(view_number):
            view_data = X[v_num][0][0].transpose()
            X_train_temp.append(view_data[all_length + train_indices])
            X_test_temp.append(view_data[all_length + test_indices])

        # Concatenate with previous classes
        if c_num == 1:
            X_train = X_train_temp
            X_test = X_test_temp
        else:
            for v_num in range(view_number):
                X_train[v_num] = np.concatenate(
                    [X_train[v_num], X_train_temp[v_num]], axis=0
                )
                X_test[v_num] = np.concatenate(
                    [X_test[v_num], X_test_temp[v_num]], axis=0
                )

        all_length += c_length

    # Normalize data if requested
    if normalize_data:
        for v_num in range(view_number):
            X_train[v_num] = normalize(X_train[v_num])
            X_test[v_num] = normalize(X_test[v_num])

    if save_feature_label:
        for v_num in range(view_number):
            train_features = np.stack(X_train[v_num])
            train_labels = np.stack(labels_train)
            test_features = np.stack(X_test[v_num])
            test_labels = np.stack(labels_test)

            train_features = torch.from_numpy(train_features)
            train_labels = torch.from_numpy(train_labels)
            test_features = torch.from_numpy(test_features)
            test_labels = torch.from_numpy(test_labels)

            # Combine train and test
            combined_features = torch.cat([train_features, test_features], dim=0)
            combined_labels = torch.cat([train_labels, test_labels], dim=0)

            print(f"Combined features shape: {combined_features.shape}")
            print(f"Combined labels shape: {combined_labels.shape}")
            print(
                f"Label range: {combined_labels.min().item()} to {combined_labels.max().item()}"
            )

            torch.save(
                combined_features,
                f"dataset/embedding/mat_file/{dataset_name}_view_{v_num}_Feature.pt",
            )
            torch.save(
                combined_labels,
                f"dataset/embedding/mat_file/{dataset_name}_view_{v_num}_Label.pt",
            )
    else:
        # Create PyTorch datasets
        train_dataset = MultiViewDataset(X_train, view_number, np.array(labels_train))
        test_dataset = MultiViewDataset(X_test, view_number, np.array(labels_test))

        return train_dataset, test_dataset, view_number


def xavier_init_pytorch(fan_in, fan_out, constant=1):
    """
    PyTorch version of Xavier initialization
    Args:
        fan_in: Number of input units
        fan_out: Number of output units
        constant: Scaling constant
    Returns:
        Initialized tensor
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return torch.empty(fan_in, fan_out).uniform_(low, high)
