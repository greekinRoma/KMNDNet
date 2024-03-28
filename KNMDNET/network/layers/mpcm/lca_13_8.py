import torch
from torch import nn
import numpy as np
from ...network_blocks import BaseConv
from setting.read_setting import config as cfg
class ExpansionContrastModule(nn.Module):
    def __init__(self,in_channels,shifts=[1,3]):
        super().__init__()
        self.shifts=shifts
        w1=np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, -1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, -1], [0, 1, 0], [0, 0, 0]],[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]])
        w1=w1.reshape(4,1,3,3)
        w2=w1[:,:,::-1,::-1].copy()
        if cfg.use_cuda:
            self.kernel1=torch.Tensor(w1).cuda()
            self.kernel2=torch.Tensor(w2).cuda()
        else:
            self.kernel1 = torch.Tensor(w1)
            self.kernel2 = torch.Tensor(w2)
        self.in_channels = in_channels//8
        self.kernel1=self.kernel1.repeat(self.in_channels,1,1,1).contiguous()
        self.kernel2=self.kernel2.repeat(self.in_channels,1,1,1).contiguous()
        self.act=torch.nn.Sigmoid()
        self.in_conv=nn.Conv2d(in_channels=in_channels,out_channels=self.in_channels,kernel_size=1,stride=1)
        self.out_conv=nn.Sequential(
            BaseConv(in_channels=self.in_channels,out_channels=self.in_channels,ksize=1,stride=1),
            nn.Conv2d(in_channels=self.in_channels,out_channels=1,kernel_size=1,stride=1)
        )
        self.mas_conv=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=1,stride=1),
            nn.Sigmoid()
        )
        self.scale1 = nn.Parameter(torch.zeros(3))
        self.scale2 = nn.Parameter(torch.zeros(3))
        self.scale3 = nn.Parameter(torch.zeros(4))
    def circ_shift(self,cen,index,shift):
        batch_size,num_channels,height,widht=cen.shape
        out1=torch.nn.functional.conv2d(weight=self.kernel1,stride=1,padding="same",dilation=shift,input=cen,groups=self.in_channels)
        out2=torch.nn.functional.conv2d(weight=self.kernel2,stride=1,padding="same",dilation=shift,input=cen,groups=self.in_channels)
        out=out1*out2
        out=out.view(batch_size,num_channels,4,height,widht).contiguous()
        scale1 = torch.softmax(self.scale1, -1)
        out = torch.min(out, dim=2, keepdim=False).values * scale1[0] + torch.mean(out, dim=2, keepdim=False) * scale1[1] \
              + torch.max(out, dim=2, keepdim=False).values * scale1[2]
        return out
    def spatial_attention(self,cen):
        outs=[]
        cen=self.in_conv(cen)
        for index,shift in enumerate(self.shifts):
            outs.append(self.circ_shift(cen,index,shift))
        outs=torch.stack(outs,dim=-1)
        scale2 = torch.softmax(self.scale2, dim=-1)
        outs = torch.min(outs, dim=-1, keepdim=False).values * scale2[0] + torch.mean(outs, dim=-1, keepdim=False) * scale2[1] \
               + torch.max(outs, dim=-1, keepdim=False).values * scale2[2]
        outs=self.out_conv(outs)
        out=torch.sigmoid(outs)
        return out
    def forward(self,cen,mas):
        out_mask=self.spatial_attention(cen)
        mas_mask=self.mas_conv(mas)
        scale3=torch.softmax(self.scale3,dim=-1)
        return cen*(out_mask*scale3[0]+mas_mask*scale3[1]+out_mask*mas_mask*scale3[2]+scale3[3])