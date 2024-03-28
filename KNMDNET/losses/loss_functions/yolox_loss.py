import PIL.Image
from losses.assignment.simota import simOTA
from losses.loss_funs.iouloss import IOUloss
import torch
from torch import nn
from VisionTools.show_ota import show_ota
from torchvision import transforms
class YOLOXLoss(nn.Module):
    def __init__(self,mode='OTA',use_cuda=True):
        super(YOLOXLoss, self).__init__()
        self.get_assignment(mode,use_cuda)
        self.iouloss=IOUloss("none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.use_l1=False
        self.l1loss=nn.L1Loss(reduction="none")
        self.use_cuda=use_cuda
        self.background_mask_resizes=[]
        self.target_mask_resizes=[]
        self.sizes=[80,40,20]
        for size in [80,40,20]:
            self.background_mask_resizes.append(transforms.Resize([size,size],interpolation=PIL.Image.NEAREST))
    def get_assignment(self,mode,use_cuda):
        if mode=='OTA':
            self.assignment=simOTA()
    def get_l1_target(self, l1_target, target, stride, grid, eps=1e-8):
        l1_target[:, 0] = target[:, 0] / stride - grid[...,0]
        l1_target[:, 1] = target[:, 1] / stride - grid[...,1]
        l1_target[:, 2] = torch.log(target[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(target[:, 3] / stride + eps)
        return l1_target
    def forward(self,targets,strides,grids,outputs,regs,masks,use_augs,imgs=None):
        num_f=0.
        num_g=0
        num_m=0
        total_loss_l1=0
        total_loss_iou=0
        total_loss_obj=0
        total_loss_cls=0
        total_loss_mask=0
        loss_list=[]
        tmp_masks=[]
        for bm_resize in self.background_mask_resizes:
            batch_size, w, h, channel = masks.shape
            bm=masks[..., 0]
            tm=masks[...,1]
            bm=bm_resize(bm)
            tm=bm_resize(tm)
            bm = bm.view(batch_size, -1, 1)
            tm=tm.view(batch_size,-1,1)
            tmp_masks.append(torch.concatenate([bm,tm],-1))
        masks=torch.concatenate(tmp_masks,-2)
        for i,(target,output,reg,mask,use_aug) in enumerate(zip(targets,outputs,regs,masks,use_augs)):
            target=target[target[...,0]>0]
            if len(target)==0:
                loss_list.append(-1)
                continue
            with torch.cuda.amp.autocast(enabled=False):
                fg_mask,pred_ious_this_matching,matched_gt_inds,num_fg,num_gt=self.assignment(output,target,grids,strides)
            if imgs is not None:
                show_ota(imgs[i], output, target, strides, grids, fg_mask, matched_gt_inds)
            loss_iou = 5. * self.iouloss(output[fg_mask,:4], target[matched_gt_inds,1:]).sum()
            loss_obj = self.bcewithlog_loss(output[:,5], fg_mask.to(output.dtype)).sum()
            loss_cls = self.bcewithlog_loss(output[fg_mask,6], pred_ious_this_matching).sum()
            index_mask=torch.sum(mask,-1)>0
            loss_mask = self.bcewithlog_loss(output[index_mask,4],mask[index_mask,1].to(output.dtype)).sum()
            num_m=num_m+torch.sum(index_mask)
            total_loss_iou = total_loss_iou+loss_iou
            total_loss_obj = total_loss_obj+loss_obj
            total_loss_cls = total_loss_cls+loss_cls
            total_loss_mask = total_loss_mask+loss_mask
            num_f = num_f + num_fg
            num_g = num_g + num_gt
            num_fg=max(num_fg,1)
            loss_list.append(((loss_obj+loss_cls)/num_fg).detach())
        num_f=max(num_f,1)
        num_g=max(num_g,1)
        l1_loss=total_loss_l1/num_f
        iou_loss=total_loss_iou/num_f
        obj_loss=total_loss_obj/num_f
        cls_loss=total_loss_cls/num_f
        mask_loss=total_loss_mask/num_m*2
        sum_loss=l1_loss+iou_loss+obj_loss+cls_loss+mask_loss
        outputs = {
            "total_loss": sum_loss,
            "iou_loss": iou_loss,
            "obj_loss": obj_loss,
            "l1_loss":l1_loss,
            "cls_loss":cls_loss,
            "mask_loss":mask_loss,
            "num_fg": num_f/num_g}
        return outputs,loss_list