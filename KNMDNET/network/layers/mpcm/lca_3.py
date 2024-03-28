from torch import nn
import torch
from ...network_blocks import BaseConv
class LCAModule(nn.Module):
    def __init__(self,channels,shift):
        super().__init__()
        self.channels=channels
        self.shift=shift
        self.conv=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=2*shift+1,padding=shift,groups=channels,bias=False)
        self.out_conv=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,bias=False)
        self.max_pool=torch.nn.AdaptiveMaxPool2d(output_size=1)
        self.channel_attention=nn.Sequential(*[nn.Conv2d(in_channels=channels,out_channels=channels//2,kernel_size=1,bias=False),
                                               nn.Conv2d(in_channels=channels//2,out_channels=channels,kernel_size=1,bias=False),
                                               nn.Softmax()])
    def forward(self,cen):
        out=self.conv(cen)
        center_weight=torch.sum(self.out_conv.weight,dim=[2,3],keepdim=True)
        max_pool=self.max_pool(out)
        channel_attention=self.channel_attention(max_pool)

        out=self.out_conv(out)
        return out
class LCA3(nn.Module):
    def __init__(self, in_channels, shifts=[3, 5, 9]):
        super().__init__()
        self.channels=in_channels
        self.conv1=torch.nn.Sequential(*[BaseConv(in_channels=in_channels, out_channels=in_channels // 2, ksize=1, stride=1, bias=False),
                                         LCAModule(channels=in_channels // 2, shift=shifts[0])])
        self.conv2=torch.nn.Sequential(*[BaseConv(in_channels=int(in_channels / 2), out_channels=in_channels // 4, ksize=1, stride=1, bias=False),
                                         LCAModule(channels=in_channels // 4, shift=shifts[1])])
        self.conv3=torch.nn.Sequential(*[BaseConv(in_channels=int(in_channels * 3 / 4), out_channels=in_channels // 4, ksize=1, stride=1, bias=False),
                                         LCAModule(channels=in_channels // 4, shift=shifts[2])])
        self.act=nn.Sequential(*[nn.BatchNorm2d(in_channels),
                                 nn.SiLU()])
    def forward(self,input):
        out1=self.conv1(input)
        out2=self.conv2(out1)
        out2=torch.concat([out1,out2],dim=1)
        out3=self.conv3(out2)
        out=torch.concat([out2,out3],dim=1)
        out=self.act(out+input)
        return out