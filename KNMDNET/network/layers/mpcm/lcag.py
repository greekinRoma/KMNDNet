from torch import nn
import torch
from network.network_blocks import BaseConv
class MCAF(nn.Module):
    def __init__(self,channels,shifts=[1,3,5,9],r=0.5):
        super().__init__()
        self.channels=channels
        self.shifts=shifts
        hidden_channels=int(r*channels)
        self.malc1=MALC(channels=channels,shifts=shifts)
        self.malc2=MALC(channels=channels,shifts=shifts)
        self.maxpool=Max_channel(channels=channels)
        self.pix_conv=nn.Sequential(*[
            torch.nn.Conv2d(in_channels=channels,out_channels=hidden_channels,kernel_size=1,stride=1),
            torch.nn.Conv2d(in_channels=hidden_channels,out_channels=channels,kernel_size=1,stride=1),
            torch.nn.Sigmoid()
        ])
    def forward(self,hcen,lcen):
        hcen=self.malc1(hcen)
        hmask=self.pix_conv(hcen)
        lcen=self.malc2(lcen)
        lcen_m=self.maxpool(lcen)
        out=hmask*lcen_m+lcen
        return out
class MALC(nn.Module):
    def __init__(self,channels,shifts):
        super().__init__()
        self.channels=channels
        self.shifts=shifts
        self.dconvs=torch.nn.ModuleList()
        self.oconvs=torch.nn.ModuleList()
        for shift in shifts:
            self.dconvs.append(torch.nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,dilation=shift//2+1,padding=shift//2+1))
            self.oconvs.append(nn.Sequential(*[torch.nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,dilation=shift//2+1,padding=shift//2+1),
                                               torch.nn.Sigmoid()]))
    def dalc(self,cen,index):
        out1=cen-self.dconvs[index](cen)
        out2=self.oconvs[index](out1)+1e-5
        out=out1/out2
        return out
    def forward(self,cen):
        outs=[]
        for index,shift in enumerate(self.shifts):
            outs.append(self.dalc(cen,index))
        outs=torch.stack(outs,dim=-1)
        outs=torch.mean(outs,dim=-1)
        out_mask=torch.sigmoid(outs)
        return out_mask*cen+cen
class Max_channel(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv=torch.nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1)
        self.max_pool=torch.nn.AdaptiveMaxPool2d(output_size=1)
    def forward(self,cen):
        out=torch.sigmoid(cen)
        out=self.max_pool(out)
        out=self.conv(out)
        return cen*out