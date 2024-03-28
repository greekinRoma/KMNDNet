from torch import nn
import torch
class soft_channels(nn.Module):
    def __init__(self,
                 in_channels:int,
                 expansion:float=0.5):
        super().__init__()
        self.in_channels=in_channels
        self.expansion=expansion
        self.max_pooling=nn.AdaptiveMaxPool2d(output_size=1)
        hidden_channels=int(expansion*self.in_channels)
        self.out_conv=nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels,out_channels=hidden_channels,kernel_size=1,stride=1),
            nn.Conv2d(in_channels=hidden_channels,out_channels=in_channels,kernel_size=1,stride=1),
            nn.Sigmoid()
        ])
    def forward(self,inp):
        out=self.max_pooling(inp)
        channel_attention=self.out_conv(out)
        return channel_attention
