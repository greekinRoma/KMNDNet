import torch
import torch.nn.functional as F
from utils import bboxes_iou
from torch import nn
from utils.iou import get_iou
class TopK(nn.Module):
    def __init__(self,ioutype:str="iou"):
        super().__init__()
        self.grid=None
        self.stride=None
        self.center_radius=1.5
        self.bbox_iou=get_iou(ioutype)
    def get_geometry_constraint(self,gt):
        #计算groundtruth的中心点位置
        x_centers_per_image = ((self.grid[...,0] + 0.5) * self.stride).clone()
        y_centers_per_image = ((self.grid[...,1] + 0.5) * self.stride).clone()

        # in fixed center
        center_radius = 1.5
        #这是固定的数据的边长，就是anchor中的
        center_dist = (self.stride * center_radius).clone()
        #计算范围的方法
        gt_bboxes_per_image_l = (gt[:, 0:1]) - center_dist#
        gt_bboxes_per_image_r = (gt[:, 0:1]) + center_dist#
        gt_bboxes_per_image_t = (gt[:, 1:2]) - center_dist#
        gt_bboxes_per_image_b = (gt[:, 1:2]) + center_dist#
        #得到目标的动态范围
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        #判断中心是否在1.5r的范围内（这个步骤我在总感觉非常puzzled,如果是大目标的语，那应该怎么处理？）
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        #每个anchor对应的box的大小，这个是通过anchor来找对应
        geometry_relation = is_in_centers[:, anchor_filter]
        #只输出anchor的大小
        return anchor_filter, geometry_relation
    def simota_matching(self,cost,pair_wise_iou, num_gt, fg_mask):
        matching_matrix=torch.zeros_like(cost,dtype=torch.uint8)
        n_candidate_k=min(10,pair_wise_iou.size(1))
        topk_iou,_=torch.topk(pair_wise_iou,n_candidate_k,dim=1)
        dynamic_ks=torch.clamp(topk_iou.sum(1).int(),min=1)
        for i in range(num_gt):
            _,pos_idx=torch.topk(
                cost[i],k=dynamic_ks[i],largest=False
            )
            matching_matrix[i][pos_idx]=1
        del topk_iou,dynamic_ks,pos_idx
        anchor_matching_gt=matching_matrix.sum(0)

        if anchor_matching_gt.max()>1:
            multiple_match_mask=anchor_matching_gt>1
            _,cost_argmin=torch.min(cost[:,multiple_match_mask],dim=0)
            matching_matrix[:,multiple_match_mask]*=0
            matching_matrix[cost_argmin,multiple_match_mask]=1
        if matching_matrix.sum(0).max()>1:
            assert True,"error matching problem"
            raise
        fg_mask_inbox=anchor_matching_gt>0
        num_fg=fg_mask_inbox.sum().item()
        fg_mask[fg_mask.clone()]=fg_mask_inbox
        matched_gt_inds = matching_matrix[:, fg_mask_inbox].argmax(0)
        pred_ious_this_matching = (matching_matrix * pair_wise_iou).sum(0)[fg_mask_inbox]
        return num_fg, pred_ious_this_matching, matched_gt_inds
    @torch.no_grad()
    def forward(self,output,target,grid,stride):
        if (self.grid is None or self.stride is None or self.grid.shape!=grid.shape or self.stride.shape!=stride.shape):
            self.grid=grid
            self.stride=stride
        box_gt=target[:,1:]
        fg_mask,gemetry_relation=self.get_geometry_constraint(box_gt)
        box_pred=output[:,:4][fg_mask]
        obj_pred=output[:,4][fg_mask]
        cls_pred=output[:,5][fg_mask]
        mask_pred=output[:,6][fg_mask]
        num_in_box_anchor=box_pred.shape[0]
        pair_wise_iou=bboxes_iou(box_gt,box_pred,False)
        pair_wise_iou_loss = -torch.log(pair_wise_iou+ 1e-8)
        num_gt=target.size(0)
        with torch.cuda.amp.autocast(enabled=False):
            cls_pred=(cls_pred.sigmoid()*obj_pred.sigmoid()).sqrt()*mask_pred.sigmoid()
            pair_wise_cls_loss=F.binary_cross_entropy(
                cls_pred.unsqueeze(0).repeat(num_gt,1),
                torch.ones([num_gt,num_in_box_anchor]).cuda(),
                reduction="none"
            )
        cost=(pair_wise_cls_loss+3.0*pair_wise_iou_loss+float(1e6)*(~gemetry_relation))
        num_fg, pred_ious_this_matching, matched_gt_inds= self.simota_matching(cost,pair_wise_iou, num_gt, fg_mask)
        del pair_wise_iou_loss,pair_wise_cls_loss,cost,pair_wise_iou
        return fg_mask,pred_ious_this_matching,matched_gt_inds,num_fg,num_gt