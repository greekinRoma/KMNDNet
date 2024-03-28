import torch
import torch.nn as nn
class IOU(nn.Module):
    def __init__(self, eps=1e-14):
        super(IOU, self).__init__()
        self.eps=eps
    def get_iou(self, pred, target):
        tl = torch.max(
            (pred[..., :2] - pred[..., 2:] / 2), (target[..., :2] - target[..., 2:] / 2)
        )
        br = torch.min(
            (pred[..., :2] + pred[..., 2:] / 2), (target[..., :2] + target[..., 2:] / 2)
        )

        area_p = torch.prod(pred[..., 2:], -1)
        area_g = torch.prod(target[..., 2:], -1)

        en = (tl < br).type(tl.type()).prod(dim=-1)
        area_i = torch.prod(br - tl, -1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + self.eps)
        return iou