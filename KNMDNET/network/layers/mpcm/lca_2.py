import torch
from torch import nn
import numpy as np
from ...network_blocks import BaseConv
from setting.read_setting import config as cfg
class LCA2(nn.Module):
    def __init__(self,in_channels,shifts=[1,3]):
        super().__init__()
        self.shifts=shifts
        self.in_channels=in_channels
        w1=np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, -1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, -1], [0, 1, 0], [0, 0, 0]],[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]])
        w1=w1.reshape(4,1,3,3)
        w2=w1[:,:,::-1,::-1].copy()
        w1=torch.Tensor(w1)
        w2=torch.Tensor(w2)
        w=torch.cat([w1,w2],dim=0)
        w=w.repeat(self.in_channels,1,1,1)
        if cfg.use_cuda:
            self.kernel=w.cuda()
        else:
            self.kernel=w
        self.out_conv=nn.Sequential(*[nn.Conv2d(in_channels=in_channels*8,out_channels=in_channels,kernel_size=1,stride=1,padding="same",bias=False,groups=in_channels),
                                      nn.BatchNorm2d(in_channels),
                                      nn.SiLU(),
                                      nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding="same",bias=False),
                                      nn.Sigmoid()])
    @torch.no_grad()
    def circ_shift(self,cen,index,shift):
        out=torch.nn.functional.conv2d(weight=self.kernel,stride=1,padding="same",dilation=shift,input=cen,groups=self.in_channels,bias=None)
        return out
    def spatial_attention(self,cen):
        out=0
        for index,shift in enumerate(self.shifts):
            out=out+self.circ_shift(cen,index,shift)
        outs=out/3
        out=self.out_conv(outs)
        # import numpy as np
        # import cv2
        # cv2.imshow("outcome",np.array(torch.max(out,dim=1).values.squeeze().cpu().detach()))
        # cv2.waitKey(0)
        return out
    def forward(self,cen,mask):
        # print(mask.shape)
        # import cv2
        # import numpy as np
        # cv2.imshow("outcome",np.array(torch.mean(mask,dim=1).squeeze().cpu().detach()))
        # cv2.waitKey(0)
        return cen*self.spatial_attention(cen)+cen