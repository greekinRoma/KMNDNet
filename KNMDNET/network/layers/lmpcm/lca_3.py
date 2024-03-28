import torch
from torch import nn
import numpy as np
from ...network_blocks import BaseConv
from setting.read_setting import config as cfg
class ExpansionContrastModule(nn.Module):
    def __init__(self,in_channels,shifts=[1,3,5,7]):
        super().__init__()
        self.convs_list=nn.ModuleList()
        self.out_convs_list=nn.ModuleList()
        self.avepools_list=nn.ModuleList()
        self.layer1_list=nn.ModuleList()
        self.layer2_list=nn.ModuleList()
        self.scale_list=nn.ModuleList()
        self.down_list=nn.ModuleList()
        self.shifts=shifts
        tmp1=np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, -1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, -1], [0, 1, 0], [0, 0, 0]],[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]])
        tmp1=tmp1.reshape(4,1,3,3)
        tmp2=tmp1[:,:,::-1,::-1].copy()
        w1=tmp1
        w2=tmp2
        self.in_channels = in_channels // 4
        self.expand_rate = 4
        w4=torch.zeros([self.in_channels,self.expand_rate,1,1])
        if cfg.use_cuda:
            self.kernel1=torch.Tensor(w1).cuda()
            self.kernel2=torch.Tensor(w2).cuda()
            self.scales1=nn.Parameter(torch.zeros(4).cuda())
            self.scales3=nn.Parameter(w4.cuda())
        else:
            self.kernel1 = torch.Tensor(w1)
            self.kernel2 = torch.Tensor(w2)
            self.scales1=torch.nn.Parameter(torch.zeros(4))
            self.scales3=nn.Parameter(w4)
        self.kernel1=self.kernel1.repeat(self.in_channels,1,1,1).contiguous()
        self.kernel2=self.kernel2.repeat(self.in_channels,1,1,1).contiguous()
        self.act=torch.nn.Sigmoid()
        self.in_conv=nn.Conv2d(in_channels=in_channels,out_channels=self.in_channels,kernel_size=1,stride=1)
        self.out_conv=nn.Sequential(
            BaseConv(in_channels=self.in_channels,out_channels=self.in_channels,ksize=1,stride=1),
            nn.Conv2d(in_channels=self.in_channels,out_channels=1,kernel_size=1,stride=1)
        )
        for shift in shifts:
            self.convs_list.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=shift,stride=1,padding=shift//2))
    def circ_shift(self,cen,index,shift):
        cen=self.convs_list[index](cen)
        kernel4=self.scales3
        out1=torch.nn.functional.conv2d(weight=self.kernel1,stride=1,padding="same",dilation=shift,input=cen,groups=self.in_channels)
        out2=torch.nn.functional.conv2d(weight=self.kernel2,stride=1,padding="same",dilation=shift,input=cen,groups=self.in_channels)
        out=out1*out2
        b, c, w, h = out.shape
        out = out.view(b, c //4, 4, w, h)
        out = torch.sort(out, dim=2).values
        out = out.view(b, c, w, h)
        out = torch.nn.functional.conv2d(weight=kernel4,stride=1,input=out,groups=self.in_channels)
        return out
    def spatial_attention(self,cen):
        outs=[]
        cen=self.in_conv(cen)
        for index,shift in enumerate(self.shifts):
            outs.append(self.circ_shift(cen,index,shift))
        outs=torch.stack(outs,dim=-1)
        outs=torch.max(outs,dim=-1,keepdim=False).values+torch.mean(outs,dim=-1,keepdim=False)
        outs=torch.relu(outs)
        outs=self.out_conv(outs)
        out=self.act(outs)
        return out
    def forward(self,cen,mas):
        out_mask=self.spatial_attention(cen)
        return cen*(out_mask+1)