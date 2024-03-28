from torch import nn
import torch
from ..network_blocks import BaseConv
class shar_layer(nn.Module):
    def __init__(self,expansion,in_channels,out_channels,act):
        super().__init__()
        hidden_channels=int(expansion*out_channels)
        self.conv1_1=BaseConv(in_channels,out_channels, ksize=1, stride=1, act=act)
        self.conv1_2=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.act=torch.nn.SiLU()
    def forward(self,inp):
        x1=self.conv1_1(inp)
        x2=self.conv1_2(inp)
        x1=self.conv2(x1)
        return self.act(x1+x2)