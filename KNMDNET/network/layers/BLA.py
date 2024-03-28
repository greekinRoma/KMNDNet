from torch import nn
import torch
from network.layers.MLC import MLC
from network.network_blocks import BaseConv
class BLA(nn.Module):
    def __init__(self,in_channel,expansion,act="silu"):
        super().__init__()
        hidden_channel=int(in_channel*expansion)
        self.expansion=expansion
        self.stem=torch.nn.Sequential(*[
            BaseConv(in_channels=in_channel,out_channels=hidden_channel,act=act,ksize=1,stride=1),
            nn.Conv2d(in_channels=hidden_channel,out_channels=in_channel,kernel_size=1,stride=1),
            torch.nn.Sigmoid()
        ])
        self.MLC=MLC()
    def forward(self,depth_layer,shallow_layer):
        m_1=self.MLC(shallow_layer)
        m_1=self.stem(m_1)
        m_2=self.MLC(shallow_layer)
        m_2=m_2*m_1
        m_3=self.MLC(depth_layer)
        m_3=m_2+m_3
        return m_3