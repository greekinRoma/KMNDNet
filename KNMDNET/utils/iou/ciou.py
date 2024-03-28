from torch import nn
import torch
import math
class CIOU(nn.Module):
    def __init__(self,eps=1e-7):
        super().__init__()
        self.eps=eps
    def get_iou(self,pred,target):
        """
                pred:x,y,w,h
                target:x,y,w,h
                """
        #compute overlap
        tl = torch.max(
            (pred[..., :2] - pred[..., 2:] / 2), (target[..., :2] - target[..., 2:] / 2)
        )
        br = torch.min(
            (pred[..., :2] + pred[..., 2:] / 2), (target[..., :2] + target[..., 2:] / 2)
        )
        wh = (br - tl).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        # union
        area_p = torch.prod(pred[..., 2:], -1)  # 计算预测结果大小
        area_g = torch.prod(target[..., 2:], -1)  # 计算GT面积大小
        union = area_p + area_g - overlap + self.eps
        # iou
        ious = overlap / union
        # enclose area
        c_tl = torch.min((pred[..., :2] - pred[..., 2:] / 2), (target[..., :2] - target[..., 2:] / 2))  # 计算最大框的tl
        c_br = torch.max((pred[..., :2] + pred[..., 2:] / 2), (target[..., :2] + target[..., 2:] / 2))  # 计算最大框的br
        c_wh = (c_br - c_tl).clamp(min=0)
        c_w = c_wh[..., 0]
        c_h = c_wh[..., 1]
        c2 = c_w ** 2 + c_h ** 2 + self.eps
        #
        left = (pred[..., 0] - target[..., 0]) ** 2
        right = (pred[..., 1] - target[..., 1]) ** 2
        rho2 = left + right
        # 计算增加长宽比
        factor = 4 / math.pi ** 2
        v = factor * torch.pow(torch.atan(target[..., 2] / target[..., 3]) - torch.atan(pred[..., 2] / pred[..., 3]), 2)
        with torch.no_grad():
            alpha = (ious > 0.5).float() * v / (1 - ious + v)
        # CIOU
        ciou = ious - (rho2 / c2 )
        return ciou.clamp(min=-1.,max=1.)