import torch
import torch.nn as nn


class SiameseNetModel(nn.Module):
    def __init__(self, architecture, input_dim):
        super(SiameseNetModel, self).__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList()

        current_dim = input_dim
        for layer in self.architecture:
            next_dim = layer
            self.layers.append(
                nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
            )
            current_dim = next_dim

    def single_forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x1, x2):
        out1 = self.single_forward(x1)
        out2 = self.single_forward(x2)

        return out1, out2
