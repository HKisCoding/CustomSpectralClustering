import torch


class BackboneModel:
    def __init__(self, model_name: str):
        self.backbone = None
        self.model_name = model_name

    def get_extractor(self):
        if self.backbone is None:
            raise ValueError("Backbone model is not initialized")
        return torch.nn.Sequential(*list(self.backbone.children())[:-1])
