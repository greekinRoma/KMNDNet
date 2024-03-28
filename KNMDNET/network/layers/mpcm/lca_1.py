import torch
from torch import nn
import numpy as np
from ...network_blocks import BaseConv
from setting.read_setting import config as cfg
class LCA1(nn.Module):
    def __init__(self,in_channels,shifts=[1,3]):
        super().__init__()
        self.shifts=shifts
        self.in_channels=in_channels
        w1=np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, -1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, -1], [0, 1, 0], [0, 0, 0]],[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]])
        w1=w1.reshape(4,1,3,3)
        w2=w1[:,:,::-1,::-1].copy()
        w1=torch.Tensor(w1)
        w2=torch.Tensor(w2)
        w1=w1.repeat(self.in_channels,1,1,1)
        w2=w2.repeat(self.in_channels,1,1,1)
        self.num_shifts=len(shifts)
        if cfg.use_cuda:
            self.kernel1=torch.Tensor(w1).cuda()
            self.kernel2=torch.Tensor(w2).cuda()
        else:
            self.kernel1 = torch.Tensor(w1)
            self.kernel2 = torch.Tensor(w2)
        self.out_conv=nn.Sequential(*[nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding="same",bias=False,groups=in_channels),
                                      nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding="same",bias=False),
                                      nn.Sigmoid()])
    @torch.no_grad()
    def circ_shift(self,cen,index,shift):
        batch_size,channels,w,h=cen.shape
        out1=torch.nn.functional.conv2d(weight=self.kernel1,stride=1,padding="same",dilation=shift,input=cen,groups=self.in_channels,bias=None)
        out2=torch.nn.functional.conv2d(weight=self.kernel2,stride=1,padding="same",dilation=shift,input=cen,groups=self.in_channels,bias=None)
        out=out1*out2
        out=out.view(batch_size,channels,4,w,h).contiguous()
        out=torch.min(out,dim=2).values
        return out
    @torch.no_grad()
    def spatial_attention(self,cen):
        out=[]
        for index,shift in enumerate(self.shifts):
            out.append(self.circ_shift(cen,index,shift))
        out=torch.stack(out,dim=-1)
        outs=torch.max(out,dim=-1).values+torch.mean(out,dim=-1)
        # import numpy as np
        # import cv2
        # cv2.imshow("outcome",np.array(torch.max(out,dim=1).values.squeeze().cpu().detach()))
        # cv2.waitKey(0)
        return outs
    def forward(self,cen,mask):
        # print(mask.shape)
        # import cv2
        # import numpy as np
        # cv2.imshow("outcome",np.array(torch.mean(mask,dim=1).squeeze().cpu().detach()))
        # cv2.waitKey(0)
        out=self.spatial_attention(cen)
        out_mask=self.out_conv(out)
        return cen*out_mask+cen