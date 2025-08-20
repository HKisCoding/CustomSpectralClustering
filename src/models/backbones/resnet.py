import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights

from src.models.backbones.base import BackboneModel


class ResNet(BackboneModel):
    def __init__(self, model_name: str, retrain=False):
        self.backbone = torch.hub.load(
            "pytorch/vision:v0.10.0", model_name, weights=ResNet18_Weights.DEFAULT
        )
        if retrain:
            self.backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.backbone.maxpool = nn.Identity()
            self.backbone.fc = nn.Identity()
