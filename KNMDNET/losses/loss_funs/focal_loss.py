from torch import nn
from torch.nn import functional as F
class FocalLoss(nn.Module):
    def __init__(self,
                 gamma:float,
                 alpha:float,
                 use_sigmoid=True):
        super().__init__()
        self.gamma=gamma
        self.alpha=alpha
        self.use_sigmoid=use_sigmoid
    def forward(self,pred,target):
        if self.use_sigmoid:
            pred_sigmoid=pred.sigmoid()
        else:
            pred_sigmoid=pred
        target=target.type_as(pred)
        pt=(1-pred_sigmoid)*target+pred_sigmoid*(1-target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        return loss