from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

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


@dataclass
class TrainingConfig(BaseConfig):
    """Training related settings"""

    lr: float = 0.002
    num_epoch: int = 100
    weight_path: str = "output"
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
    min_lr: float = 1e-6
    scale_k: int = 10
    lr_decay: float = 0.1
    patience: int = 10
    architecture: List[int] = field(default_factory=lambda: [512, 256, 128])
    is_local_scale: bool = False


@dataclass
class SelfAdjustGraphConfig(BaseConfig):
    """SelfAdjustGraph specific settings"""

    hid_unit: int = 256
    feat_size: int = 512
    out_feat: int = 128
    g_dim: int = 64
    k: int = 10
    gamma: float = 0.1
    mu: float = 0.1
    delta: float = 0.1
    eta: float = 0.1
    cluster: int = 10


@dataclass
class SCHOOLConfig(BaseConfig):
    """SCHOOL model specific settings"""

    node_size: int = 512
    gcn_hid_units: int = 256
    hid_units: int = 128
    n_neighbors: int = 10


@dataclass
class Config(BaseConfig):
    """Main configuration class that combines all config groups"""

    # All configs are optional with default factories
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    self_adjust_graph: SelfAdjustGraphConfig = field(
        default_factory=SelfAdjustGraphConfig
    )
    school: SCHOOLConfig = field(default_factory=SCHOOLConfig)

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
        }
        return config_classes.get(name)
