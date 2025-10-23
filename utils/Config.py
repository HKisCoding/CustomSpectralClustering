from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import torch


@dataclass
class BaseConfig:
    """Base configuration class that all config groups should inherit from"""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class DatasetConfig(BaseConfig):
    """Dataset related settings"""

    dataset: str = "default"
    batch_size: int = 32
    data_path: dict = field(
        default_factory=lambda: {
            "mnist": {
                "features": "dataset/embedding/auto_encoder/mnist_raw_Feature.pt",
                "labels": "dataset/embedding/auto_encoder/mnist_raw_Label.pt",
            },
            "Caltech_101": {
                "features": "dataset/embedding/resnet/Caltech_101_Feature.pt",
                "labels": "dataset/embedding/resnet/Caltech_101_Label.pt",
            },
            "MSRC-v2": {
                "features": "dataset/embedding/resnet/MSRC-v2_Feature.pt",
                "labels": "dataset/embedding/resnet/MSRC-v2_Label.pt",
            },
            "colon_cancer": {
                "path": "dataset/colon_cancer/colon_cancer.csv",
            },
            "leumika": {
                "path": "dataset/Leukemia/leukemia.csv",
            },
            "prokaryotic": {
                "path": "dataset/prokaryotic.mat",
            },
            "coil-20": {
                "features": "dataset/embedding/resnet/coil-20_Feature.pt",
                "labels": "dataset/embedding/resnet/coil-20_Label.pt",
            },
            "usps_mnist": {
                "features": "dataset/embedding/resnet/usps_mnist_Feature.pt",
                "labels": "dataset/embedding/resnet/usps_mnist_Label.pt",
            },
            "usps_mnist_ae": {
                "features": "dataset/embedding/auto_encoder/USPS_Feature.pt",
                "labels": "dataset/embedding/auto_encoder/USPS_Label.pt",
            },
            "animal-50": {
                "features": "dataset/embedding/mat_file/animal-50_view_0_Feature.pt",
                "labels": "dataset/embedding/mat_file/animal-50_view_0_Label.pt",
            },
            "fashion-mnist": {
                "features": "dataset/embedding/auto_encoder/fashion_mnist_Feature.pt",
                "labels": "dataset/embedding/auto_encoder/fashion_mnist_Label.pt",
            },
        }
    )


@dataclass
class TrainingConfig(BaseConfig):
    """Training related settings"""

    lr: float = 0.002
    num_epoch: int = 100
    weight_path: str = "output/weights"
    save_every: int = 10


@dataclass
class BackboneConfig(BaseConfig):
    """Backbone model settings for feature extraction"""

    name: str = "resnet18"
    pretrained: bool = True
    feature_dims: int = 512
    z_dims: int = 256


@dataclass
class OptimizationConfig(BaseConfig):
    """Optimization related settings"""

    eps: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 0.01
    tau: float = 0.05
    z_weight: float = 0.1


@dataclass
class SpectralConfig(BaseConfig):
    """Spectral clustering related settings"""

    n_nbg: int = 10
    min_lr: float = 1e-8
    scale_k: int = 10
    lr_decay: float = 0.1
    patience: int = 10
    architecture: List[int] = field(default_factory=lambda: [512, 256, 128])
    is_local_scale: bool = False


@dataclass
class SelfAdjustGraphConfig(BaseConfig):
    """SelfAdjustGraph specific settings"""

    g_dim: int = 64
    gamma: float = 0.1
    mu: float = 0.1
    delta: float = 0.1
    eta: float = 0.1
    theta: float = 0.1
    cluster: int = 10
    auxillary_loss_kind: str = "entropy"
    auxillary_loss_alpha: float = 1.0


@dataclass
class SCHOOLConfig(BaseConfig):
    """SCHOOL model specific settings"""

    node_size: int = 512
    gcn_architecture: List[int] = field(default_factory=lambda: [512, 256])
    gae_architecture: List[int] = field(default_factory=lambda: [1024, 256, 64])
    n_neighbors: int = 10
    feat_size: int = 512
    out_feat: int = 128
    k: int = 10
    spectral_architecture: List[int] = field(default_factory=lambda: [512, 256, 128])


@dataclass
class SiameseConfig(BaseConfig):
    """Siamese network related settings"""

    lr: float = 1e-3
    n_nbg: int = 2
    min_lr: float = 1e-7
    epochs: int = 100
    lr_decay: float = 0.1
    patience: int = 10
    architecture: List[int] = field(default_factory=lambda: [1024, 1024, 512])
    batch_size: int = 128
    use_approx: bool = False


@dataclass
class AutoEncoderConfig(BaseConfig):
    """AutoEncoder model related settings"""

    lr: float = 1e-3
    n_nbg: int = 2
    min_lr: float = 1e-7
    epochs: int = 50
    lr_decay: float = 0.1
    patience: int = 10
    architecture: List[int] = field(default_factory=lambda: [512, 512, 2048, 10])
    batch_size: int = 256
    weight_path: str = "./output/weights/auto_encoder/usps_mnist.pth"


@dataclass
class AEConvConfig(BaseConfig):
    """AEConv model related settings"""

    batch_size: int = 256
    weight_path: str = "ae_conv/ae_conv.pth"
    epochs: int = 100


@dataclass
class DSCConfig(BaseConfig):
    """DSC model related settings"""

    hidden_units: int = 64
    batch_size: int = 256
    n_neighbors: int = 10
    scale_k: int = 10
    n_iter: int = 20
    ae_conv: AEConvConfig = field(default_factory=AEConvConfig)


@dataclass
class Config(BaseConfig):
    """Main configuration class that combines all config groups"""

    # All configs are optional with default factories
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    spectral: SpectralConfig | None = field(default_factory=SpectralConfig)
    self_adjust_graph: SelfAdjustGraphConfig | None = field(
        default_factory=SelfAdjustGraphConfig
    )
    school: SCHOOLConfig | None = field(default_factory=SCHOOLConfig)
    siamese: SiameseConfig | None = field(default_factory=SiameseConfig)
    auto_encoder: AutoEncoderConfig | None = field(default_factory=AutoEncoderConfig)
    dsc: DSCConfig | None = field(default_factory=DSCConfig)
    # Device settings
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    def __post_init__(self):
        # Convert any dictionary values to their respective config classes
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                config_class = self._get_config_class(key)
                if config_class:
                    setattr(self, key, config_class(**value))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary."""
        return cls(**config_dict)

    @staticmethod
    def _get_config_class(name: str) -> Optional[Type[BaseConfig]]:
        """Get the config class for a given config group name."""
        config_classes = {
            "dataset": DatasetConfig,
            "training": TrainingConfig,
            "backbone": BackboneConfig,
            "optimization": OptimizationConfig,
            "spectral": SpectralConfig,
            "self_adjust_graph": SelfAdjustGraphConfig,
            "school": SCHOOLConfig,
            "siamese": SiameseConfig,
            "auto_encoder": AutoEncoderConfig,
            "dsc": DSCConfig,
            "ae_conv": AEConvConfig,
        }
        return config_classes.get(name)
