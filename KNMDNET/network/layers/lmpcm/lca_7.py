import torch
from torch import nn
import numpy as np
from ...network_blocks import BaseConv
from setting.read_setting import config as cfg
class ExpansionContrastModule(nn.Module):
    def __init__(self,in_channels,shifts=[1,3,5,7]):
        super().__init__()
        self.convs_list=nn.ModuleList()
        self.layer1_list=nn.ModuleList()
        self.layer2_list=nn.ModuleList()
        self.shifts=shifts
        tmp1=np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, -1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, -1], [0, 1, 0], [0, 0, 0]],[[0, 0, 0], [0, 1, -1], [0, 0, 0]]])
        tmp1=tmp1.reshape(4,1,3,3)
        tmp2=tmp1[:,:,::-1,::-1].copy()
        tmp=np.concatenate([tmp1,tmp2],axis=0)
        w1,w2,w3,w4,w5,w6,w7,w8=np.array_split(tmp,8)
        self.in_channels = in_channels//4
        if cfg.use_cuda:
            self.kernel1 = torch.Tensor(w1).cuda()
            self.kernel2 = torch.Tensor(w2).cuda()
            self.kernel3 = torch.Tensor(w3).cuda()
            self.kernel4 = torch.Tensor(w4).cuda()
            self.kernel5 = torch.Tensor(w5).cuda()
            self.kernel6 = torch.Tensor(w6).cuda()
            self.kernel7 = torch.Tensor(w7).cuda()
            self.kernel8 = torch.Tensor(w8).cuda()
        else:
            self.kernel1 = torch.Tensor(w1)
            self.kernel2 = torch.Tensor(w2)
            self.kernel3 = torch.Tensor(w3)
            self.kernel4 = torch.Tensor(w4)
            self.kernel5 = torch.Tensor(w5)
            self.kernel6 = torch.Tensor(w6)
            self.kernel7 = torch.Tensor(w7)
            self.kernel8 = torch.Tensor(w8)
        self.kernel1 = self.kernel1.repeat(self.in_channels,1,1,1).contiguous()
        self.kernel2 = self.kernel2.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel3 = self.kernel3.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel4 = self.kernel4.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel5 = self.kernel5.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel6 = self.kernel6.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel7 = self.kernel7.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel8 = self.kernel8.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.res_block = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                                   padding=1)
        self.in_conv=nn.Conv2d(in_channels=in_channels,out_channels=self.in_channels,kernel_size=1,stride=1)
        self.out_conv=nn.Sequential(
            BaseConv(in_channels=self.in_channels*2,out_channels=self.in_channels,ksize=1,stride=1,groups=self.in_channels),
            nn.Conv2d(in_channels=self.in_channels,out_channels=1,kernel_size=1,stride=1),
            nn.Sigmoid()
        )
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1)
        self.act_layer=torch.nn.SiLU()
        for shift in shifts:
            self.convs_list.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=shift,stride=1,padding=shift//2,groups=self.in_channels))
            self.layer1_list.append(nn.Conv2d(in_channels=self.in_channels*4,out_channels=self.in_channels,kernel_size=1,stride=1,groups=self.in_channels))
            self.layer2_list.append(nn.Conv2d(in_channels=self.in_channels*8,out_channels=self.in_channels,kernel_size=1,stride=1,groups=self.in_channels))
    def circ_shift(self,cen,index,shift):
        b,c,w,h=cen.shape
        cen=self.convs_list[index](cen)
        tmp1 = torch.nn.functional.conv2d(weight=self.kernel1, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        tmp2 = torch.nn.functional.conv2d(weight=self.kernel2, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        tmp3 = torch.nn.functional.conv2d(weight=self.kernel3, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        tmp4 = torch.nn.functional.conv2d(weight=self.kernel4, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        tmp5 = torch.nn.functional.conv2d(weight=self.kernel5, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        tmp6 = torch.nn.functional.conv2d(weight=self.kernel6, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        tmp7 = torch.nn.functional.conv2d(weight=self.kernel7, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        tmp8 = torch.nn.functional.conv2d(weight=self.kernel8, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        out1 = torch.stack([tmp2 + tmp6 , tmp4 + tmp8, tmp3 + tmp7 , 2 * tmp5], dim=2).view(b, -1, w, h)
        out2 = torch.stack([tmp3 + tmp7 , tmp5 + tmp1, tmp4 + tmp8 , 2 * tmp6], dim=2).view(b, -1, w, h)
        out3 = torch.stack([tmp4 + tmp8 , tmp6 + tmp2, tmp5 + tmp1 , 2 * tmp7], dim=2).view(b, -1, w, h)
        out4 = torch.stack([tmp5 + tmp1 , tmp7 + tmp3, tmp6 + tmp2 , 2 * tmp8], dim=2).view(b, -1, w, h)
        out5 = torch.stack([tmp6 + tmp2 , tmp8 + tmp4, tmp7 + tmp3 , 2 * tmp1], dim=2).view(b, -1, w, h)
        out6 = torch.stack([tmp7 + tmp3 , tmp1 + tmp5, tmp8 + tmp4 , 2 * tmp2], dim=2).view(b, -1, w, h)
        out7 = torch.stack([tmp8 + tmp4 , tmp2 + tmp6, tmp1 + tmp5 , 2 * tmp3], dim=2).view(b, -1, w, h)
        out8 = torch.stack([tmp1 + tmp5 , tmp3 + tmp7, tmp2 + tmp6 , 2 * tmp4], dim=2).view(b, -1, w, h)
        out1 = self.layer1_list[index](out1)*tmp1
        out2 = self.layer1_list[index](out2)*tmp2
        out3 = self.layer1_list[index](out3)*tmp3
        out4 = self.layer1_list[index](out4)*tmp4
        out5 = self.layer1_list[index](out5)*tmp5
        out6 = self.layer1_list[index](out6)*tmp6
        out7 = self.layer1_list[index](out7)*tmp7
        out8 = self.layer1_list[index](out8)*tmp8
        out = torch.stack([out1,out2,out3,out4,out5,out6,out7,out8],dim=2)
        out = torch.sort(out, dim=2).values
        out = out.view(b, -1, w, h)
        out = self.layer2_list[index](out)
        return out
    def spatial_attention(self,cen):
        outs=[]
        cen=self.in_conv(cen)
        for index,shift in enumerate(self.shifts):
            outs.append(self.circ_shift(cen,index,shift))
        outs = torch.stack(outs, dim=2)
        outs = torch.sort(outs,dim=2).values
        b, c, d, w, h = outs.shape
        outs = outs.view(b, -1, w, h).contiguous()
        outs = self.out_conv(outs)
        return outs
    def forward(self,cen,mas):
        outs = self.spatial_attention(cen)
        return outs