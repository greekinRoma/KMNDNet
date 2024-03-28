from torch import nn
import torch
def _nms(heat, kernel=3):
    """
    input:heat:热力学图，即centernet使用的heatmap;kernel:int,没什么用
    output:heat*keep寻找代码中的极值点
    func:保留kernel大小为3范围内的极大值
    """
    pad = (kernel - 1) // 2#计算扩张，计算padding的操作
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)#计算热力学图
    keep = (hmax == heat).float()#
    return heat * keep
def _gather_feat(feat, ind, mask=None):
    """
    feat:特征图，获取特征的大小（batc,size,feature）
    ind:
    mask:掩护模
    """
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
def _topk(scores, K=40):
    """
    scores:即heatmap，也就是目标大小
    K：选取多少个pred_box用来训练
    """
    batch, cat, height, width = scores.size()
    """获取每个类别的前k个被检测到的目标"""
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)#获取heatmap中前K的最大值
    #获取前K的score的dot的位置(x,y)的位置
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    """获取类别之间的前50，同时获取目标的种类"""
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    """找到对应的x和y的位置"""
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)#获取每个位置点的具体
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat