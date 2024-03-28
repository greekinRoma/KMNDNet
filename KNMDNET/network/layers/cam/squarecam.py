import torch
from torch import nn
import numpy as np
from setting.read_setting import config as cfg
from network.network_blocks import BaseConv
import math
class STConv(nn.Module):
    def __init__(self,in_channels,out):
        super().__init__()
        self.convs_list=nn.ModuleList()
        self.out=out
        delta1=np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, -1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, -1], [0, 1, 0], [0, 0, 0]],[[0, 0, 0], [0, 1, -1], [0, 0, 0]]])
        delta1=delta1.reshape(4,1,3,3)
        delta2=delta1[:,:,::-1,::-1].copy()
        delta=np.concatenate([delta1,delta2],axis=0)
        w1,w2,w3,w4,w5,w6,w7,w8=np.array_split(delta,8)
        self.in_channels = max(in_channels//2,1)
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
        self.kernel1 = self.kernel1.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel2 = self.kernel2.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel3 = self.kernel3.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel4 = self.kernel4.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel5 = self.kernel5.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel6 = self.kernel6.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel7 = self.kernel7.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel8 = self.kernel8.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.out_conv=nn.Conv2d(in_channels=in_channels*3,out_channels=out,kernel_size=1,stride=1)
        self.inp_layer=nn.Sequential(*[nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=3,stride=2,padding=1),
                                       nn.SiLU()])
        self.layer1=nn.Conv2d(in_channels=self.in_channels*8,out_channels=self.in_channels*2,kernel_size=1,stride=1,groups=self.in_channels,bias=False)
        self.layer2=nn.Conv2d(in_channels=self.in_channels*8,out_channels=self.in_channels*2,kernel_size=1,stride=1,groups=self.in_channels,bias=False)
        self.layer3=nn.Conv2d(in_channels=self.in_channels*8,out_channels=self.in_channels*2,kernel_size=1,stride=1,groups=self.in_channels,bias=False)
        self.out_layer=nn.Sequential(*[
            nn.Conv2d(in_channels=self.in_channels*4, out_channels=self.in_channels*2, kernel_size=1, stride=1,groups=self.in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.in_channels*2,out_channels=in_channels,kernel_size=1,stride=1)
        ])
    def initialize_biases(self, prior_prob):
        b = self.out_conv.bias.view(1, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.out_conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def delta_conv(self,cen):
        cen=self.inp_layer(cen)
        b, c, w, h = cen.shape
        delta1 = torch.nn.functional.conv2d(weight=self.kernel1, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=3)
        delta2 = torch.nn.functional.conv2d(weight=self.kernel2, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=3)
        delta3 = torch.nn.functional.conv2d(weight=self.kernel3, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=3)
        delta4 = torch.nn.functional.conv2d(weight=self.kernel4, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=3)
        delta5 = torch.nn.functional.conv2d(weight=self.kernel5, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=3)
        delta6 = torch.nn.functional.conv2d(weight=self.kernel6, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=3)
        delta7 = torch.nn.functional.conv2d(weight=self.kernel7, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=3)
        delta8 = torch.nn.functional.conv2d(weight=self.kernel8, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=3)
        deltas = torch.stack([delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8], dim=2).view(b,-1,w,h).contiguous()
        square = self.layer1(deltas)*self.layer2(deltas)
        deltas = self.layer3(deltas)
        return square,deltas
    def spatial_attention(self,inps,mas):
        sout,dout=self.delta_conv(mas)
        outs=torch.concat([sout,dout,inps],dim=1)
        outs = self.out_conv(outs)
        return outs
    def forward(self,cen,mas=None):
        outs=self.spatial_attention(cen,mas)
        return outs