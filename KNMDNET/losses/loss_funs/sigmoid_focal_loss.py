import torch
from torch import nn
from torch.nn import functional as F
class Sgmoid_FocalLoss(nn.Module):
    def __init__(self,gamma,alpha,use_sigmoid=True):
        super().__init__()
        self.gamma=gamma
        self.alpha=alpha
        self.use_sigmoid=use_sigmoid
    def forward(self,pred,target):
        if self.use_sigmoid:
            p=pred.sigmoid()
        else:
            p=pred
        t=target
        num_classes=pred.shape[-1]
        dtype=target.dtype
        device=target.device
        class_range=torch.arange(1,num_classes+1,dtype=dtype,device=device)
        term1 = (1 - p) ** self.gamma * torch.log(p)
        term2 = p ** self.gamma * torch.log(1 - p)
        return -(t == class_range).float() * term1 * self.alpha - \
            ((t != class_range) * (t >= 0)).float() * term2 * (1 - self.alpha)
