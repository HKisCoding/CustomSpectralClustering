import torch 
import torch.nn as nn 
import torch.nn.functional as F
from src.models.backbones.resnet import ResNet


class ManifoldEmbedingNetwork(nn.Module):
    def __init__(self, backbone_name, feature_dims, z_dims):
        super(ManifoldEmbedingNetwork, self).__init__()
        self.backbone = ResNet(model_name=backbone_name).get_backbone()
        self.pre_feature = nn.Sequential(nn.Linear(feature_dims, 4096),
                                         nn.BatchNorm1d(4096),
                                         nn.ReLU())
        self.subspace = nn.Linear(4096, z_dims)

    def _get_pre_feature(self, x: torch.Tensor, is_feature: bool):
        if is_feature:
            feature = x
        else:
            feature = self.backbone(x)
        pre_feature = self.pre_feature(feature)
        return pre_feature

    def forward(self, x: torch.Tensor, detach_feature=False, is_feature = False): 
        if detach_feature:
            with torch.no_grad():
                pre_feature = self._get_pre_feature(x, is_feature)
        else:
            pre_feature = self._get_pre_feature(x, is_feature)

        z = self.subspace(pre_feature)
        z = F.normalize(z)

        return z
    

class Gumble_Softmax(nn.Module):
    def __init__(self,tau, straight_through=False):
        super().__init__()
        self.tau = tau
        self.straight_through = straight_through
    
    def forward(self,logits):
        logps = torch.log_softmax(logits,dim=1)
        gumble = torch.rand_like(logps).log().mul(-1).log().mul(-1)
        logits = logps + gumble
        out = (logits/self.tau).softmax(dim=1)
        if not self.straight_through:
            return out
        else:
            out_binary = (logits*1e8).softmax(dim=1).detach()
            out_diff = (out_binary - out).detach()
            return out_diff + out



