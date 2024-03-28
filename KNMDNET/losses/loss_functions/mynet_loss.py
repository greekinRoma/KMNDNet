import PIL.Image
from losses.assignment.simota import simOTA
from losses.loss_funs.reg_loss import GetIouLoss
from losses.loss_funs.focal_loss import FocalLoss
import torch
from torch import nn
from VisionTools.show_ota import show_ota
from torchvision import transforms
class MyNetLoss(nn.Module):
    def __init__(self,
                 mode='OTA',
                 gamma=2,
                 alpha=0.25,
                 name="iou_loss",
                 use_cuda=True):
        super(MyNetLoss, self).__init__()
        self.get_assignment(mode,use_cuda)
        self.iouloss=GetIouLoss(name=name,reduction="none")
        self.focal_loss=FocalLoss(gamma=gamma,alpha=alpha)
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
        else:
            self.assignment=simOTA()
    def forward(self,
                targets,
                strides,
                grids,
                outputs,
                regs,
                masks,
                use_augs,
                imgs=None):
        """
        targets:gt
        strides:
        gtids：
        outputs:
        regs:
        masks:
        use_augs:
        img:
        """
        num_f=0.
        num_g=0
        num_m=0
        total_loss_iou=0
        total_loss_reg=0
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
            index_mask = torch.sum(mask, -1) > 0
            gt_classes_target=torch.zeros_like(output[:,6:])
            gt_classses=target[matched_gt_inds,0].to(torch.long)-1
            gt_classes_target[fg_mask,gt_classses]=1
            loss_reg =self.iouloss(output[fg_mask,:4], target[matched_gt_inds,1:]).sum()
            loss_mas= self.bcewithlog_loss(output[index_mask, 4], mask[index_mask, 1].to(output.dtype)).sum()
            loss_iou = self.bcewithlog_loss(output[fg_mask,5], pred_ious_this_matching).sum()
            loss_cls=self.bcewithlog_loss(output[:,6:],gt_classes_target).sum()
            num_m=num_m+torch.sum(index_mask)
            total_loss_iou = total_loss_iou+loss_iou
            total_loss_reg = total_loss_reg+loss_reg
            total_loss_cls = total_loss_cls+loss_cls
            total_loss_mask = total_loss_mask+loss_mas
            num_f = num_f + num_fg
            num_g = num_g + num_gt
        num_f=max(num_f,1)
        num_g=max(num_g,1)
        iou_loss=total_loss_iou/num_f
        reg_loss=total_loss_reg/num_f*5.
        cls_loss=total_loss_cls/num_f
        mask_loss=total_loss_mask/num_m*2.
        sum_loss=iou_loss+reg_loss+cls_loss+mask_loss
        outputs = {
            "total_loss": sum_loss,
            "iou_loss": iou_loss,
            "reg_loss": reg_loss,
            "cls_loss":cls_loss,
            "mask_loss":mask_loss,
            "num_fg": num_f/num_g}
        return outputs,loss_list