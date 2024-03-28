import torch
from torch import nn
import numpy as np
from setting.read_setting import config as cfg
from network.network_blocks import BaseConv
import math
class PlatContrastModule(nn.Module):
    def __init__(self,in_channels,out):
        super().__init__()
        self.convs_list=nn.ModuleList()
        self.out=out
        self.out_conv=nn.Conv2d(in_channels=in_channels*2,out_channels=out,kernel_size=1,stride=1)
        self.input_layer=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
        self.layer1=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
        self.layer2=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
    def initialize_biases(self, prior_prob):
        b = self.out_conv.bias.view(1, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.out_conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def delta_conv(self,cen):
        square = self.layer1(cen)*self.layer2(cen)
        return square
    def spatial_attention(self,inps):
        sout=self.delta_conv(inps)
        inps=self.input_layer(inps)
        outs=torch.concat([sout,inps],1)
        outs = self.out_conv(outs)
        return outs
    def forward(self,cen,mas=None):
        outs=self.spatial_attention(cen)
        return outs