import torch
from torch import nn
class GIOU(nn.Module):
    def __init__(self, eps=1e-14):
        super(GIOU, self).__init__()
        self.eps=eps
    def get_iou(self, pred, target):
        tl = torch.max((pred[..., :2] - pred[..., 2:] / 2), (target[..., :2] - target[..., 2:] / 2))#计算相交区域的tl值
        br = torch.min((pred[..., :2] + pred[..., 2:] / 2), (target[..., :2] + target[..., 2:] / 2))#计算相交区域的br值
        area_p = torch.prod(pred[..., 2:], -1)#计算预测结果大小
        area_g = torch.prod(target[..., 2:], -1)#计算GT面积大小
        en = (tl < br).type(tl.type()).prod(dim=-1)#确定那些地方是没有出现相交的预测输出
        area_i = torch.prod(br - tl, -1) * en#计算相交区域大小
        area_u = area_p + area_g - area_i#计算计算并区域的大小
        iou = (area_i) / (area_u + self.eps)#计算iou
        c_tl = torch.min((pred[..., :2] - pred[..., 2:] / 2), (target[..., :2] - target[..., 2:] / 2))#计算最大框的tl
        c_br = torch.max((pred[..., :2] + pred[..., 2:] / 2), (target[..., :2] + target[..., 2:] / 2))#计算最大框的br
        area_c = torch.prod(c_br - c_tl, -1)#计算最大框的面积
        giou = iou - (area_c - area_u) / area_c.clamp(1e-16)#计算差距距离
        return giou