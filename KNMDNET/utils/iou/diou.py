import torch
from torch import nn
class DIOU(nn.Module):
    def __init__(self, eps=1e-7):
        super(DIOU, self).__init__()
        self.eps=eps
    def get_iou(self, pred, target):
        """
        DOIU
        pred:x,y,w,h
        target:x,y,w,h
        """
        tl = torch.max((pred[..., :2] - pred[..., 2:] / 2), (target[..., :2] - target[..., 2:] / 2))#计算相交区域的tl值
        br = torch.min((pred[..., :2] + pred[..., 2:] / 2), (target[..., :2] + target[..., 2:] / 2))#计算相交区域的br值
        wh=(br-tl).clamp(min=0)#计算目标的长宽
        overlap=wh[...,0]*wh[...,1]#计算相交处的面积
        area_p = torch.prod(pred[..., 2:], -1)  # 计算预测结果大小
        area_g = torch.prod(target[..., 2:], -1)  # 计算GT面积大小
        union=area_p+area_g-overlap+self.eps#计算合并的面积
        ious=overlap/union#使用GIOU
        c_tl = torch.min((pred[..., :2] - pred[..., 2:] / 2), (target[..., :2] - target[..., 2:] / 2))  # 计算最大框的tl
        c_br = torch.max((pred[..., :2] + pred[..., 2:] / 2), (target[..., :2] + target[..., 2:] / 2))  # 计算最大框的br
        c_wh=(c_br-c_tl).clamp(min=0)#计算enclose的算法
        c_w=c_wh[...,0]
        c_h=c_wh[...,1]
        c2=c_w**2+c_h**2+self.eps
        #计算对角点的距离
        left=(pred[...,0]-target[...,0])**2
        right=(pred[...,1]-target[...,1])**2
        rho2=left+right
        dious=ious-rho2/c2
        #计算diou，使用非常直观的方式，添加了距离，这样可以加快收敛
        return dious