import torch
from torch import nn
from utils.centernetutils import _nms,_topk,_transpose_and_gather_feat
def ctdet_decode(output,cat_spec_wh=False, K=100):
    """
    output:网络输出（batchsize,channels(reg+cls(centernet(应该是使用的是coupled head)+mask)),w,h）
    """
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    reg=output[...,:2]
    wh=output[...,2:4]
    heat=output[...,4:-2]
    mas=output[...,-2:-1]
    batch, cat, height, width = heat.size()
    heat = _nms(heat)#将打下范围为K的最大值提取出来
    scores, inds, clses, ys, xs = _topk(heat, K=K)#获得前K大的Score
    mas=_transpose_and_gather_feat(mas,inds)
    """将前几个数据调整为reg，对reg进行解码"""
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)#找到Top50的具体位置
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)#获取top50的对应的stride
    if cat_spec_wh:#计算获得获得大小的方法
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes,scores,clses,mas], dim=2)
    return detections