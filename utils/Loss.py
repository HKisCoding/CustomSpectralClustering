import torch
import torch.nn as nn


class SpectralClusterLoss(nn.Module):
    def __init__(self):
        super(SpectralClusterLoss, self).__init__()

    def forward(self, W: torch.Tensor, Y: torch.Tensor, is_normalized: bool = False):
        """
        Args:
            W: Affinity Matrix
            Y: Spectral Net output
            is_normalized: Whether to use the Nomalized Laplacian Matrix
        """
        m = Y.size(0)
        if is_normalized:
            D = torch.sum(W, dim=1)
            Y = Y / torch.sqrt(D)[:, None]
        Dy = torch.cdist(Y, Y)
        loss = torch.sum(W * Dy.pow(2)) / m**2

        return loss 
    

class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self,X):
        return - self.compute_discrimn_loss(X.T)


class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, eps=0.01, gamma=1):
        super(MaximalCodingRateReduction, self).__init__()
        self.eps = eps
        self.gamma = gamma
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def compute_compress_loss(self, W, Pi):
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p,device=W.device).expand((k,p,p))
        trPi = Pi.sum(2) + 1e-8
        scale = (p/(trPi*self.eps)).view(k,1,1)
        
        W = W.view((1,p,m))
        log_det = torch.logdet(I + scale*W.mul(Pi).matmul(W.transpose(1,2)))
        compress_loss = (trPi.squeeze()*log_det/(2*m)).sum()
        return compress_loss
        
    def forward(self, X, Y, num_classes=None):
        #This function support Y as label integer or membership probablity.
        if len(Y.shape)==1:
            #if Y is a label vector
            if num_classes is None:
                num_classes = Y.max() + 1
            Pi = torch.zeros((num_classes,1,Y.shape[0]),device=Y.device)
            for indx, label in enumerate(Y):
                Pi[label,0,indx] = 1
        else:
            #if Y is a probility matrix
            if num_classes is None:
                num_classes = Y.shape[1]
            Pi = Y.T.reshape((num_classes,1,-1))
            
        W = X.T
        discrimn_loss = self.compute_discrimn_loss(W)
        compress_loss = self.compute_compress_loss(W, Pi)
 
        total_loss = - discrimn_loss + self.gamma*compress_loss
        return total_loss, [discrimn_loss.item(), compress_loss.item()]