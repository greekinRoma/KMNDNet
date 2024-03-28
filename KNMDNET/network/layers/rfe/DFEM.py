from torch import nn
import torch
from network.network_blocks import BaseConv
class C3block(nn.Module):
    def __init__(self,channels,d):
        super().__init__()
        padding = d
        if d == 1:
            self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False,dilation=d)
        else:
            combine_kernel = 2 * d - 1
            self.conv = nn.Sequential(*[nn.Conv2d(channels, channels, kernel_size=(combine_kernel, 1), stride=1, padding=(padding - 1, 0),groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=(1, combine_kernel), stride=1, padding=(0, padding - 1),groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 3, stride=1, padding=(padding, padding), groups=channels, bias=False,dilation=d),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)])
    def forward(self,input):
        output=self.conv(input)
        return output
class DFEM(nn.Module):
    def __init__(self,channels,shifts=[1,3,5,7]):
        super().__init__()
        self.convs_list=torch.nn.ModuleList()
        self.shifts=shifts
        for shift in shifts:
            self.convs_list.append(nn.Sequential(*[nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,dilation=shift//2+1,padding=shift//2+1),
                                                   nn.BatchNorm2d(channels),
                                                   nn.ReLU()]))
        self.out_conv=nn.Sequential(*[nn.Conv2d(in_channels=channels*len(shifts),out_channels=channels,kernel_size=1,stride=1),
                                      nn.BatchNorm2d(channels),
                                      nn.ReLU()])
        self.silu=torch.nn.SiLU()
    def forward(self,cen):
        outs=[]
        for index,shift in enumerate(self.shifts):
            outs.append(self.convs_list[index](cen))
        outs=torch.concatenate(outs,dim=1)
        outs=self.out_conv(outs)+cen
        outs=self.silu(outs)
        return outs
class DFEM_1(nn.Module):
    def __init__(self,channels,shifts=[3,5,7,9]):
        super().__init__()
        self.convs_list=torch.nn.ModuleList()
        self.shifts=shifts
        for shift in shifts:
            self.convs_list.append(nn.Sequential(*[nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,dilation=shift//2+1,padding=shift//2+1),
                                                   nn.BatchNorm2d(channels),
                                                   nn.PReLU()]))
        self.out_conv=nn.Sequential(*[nn.Conv2d(in_channels=channels*len(shifts),out_channels=channels,kernel_size=1,stride=1),
                                      nn.BatchNorm2d(channels),
                                      nn.PReLU()])
        self.silu=torch.nn.SiLU()
    def forward(self,cen):
        outs=[]
        for index,shift in enumerate(self.shifts):
            outs.append(self.convs_list[index](cen))
        outs=torch.concatenate(outs,dim=1)
        outs=self.out_conv(outs)+cen
        outs=self.silu(outs)
        return outs
class DFEM_2(nn.Module):
    def __init__(self,channels,shifts=[1,3,5,7]):
        super().__init__()
        self.convs_list=torch.nn.ModuleList()
        self.shifts=shifts
        for shift in shifts:
            self.convs_list.append(C3block(channels=channels,d=shifts[1]))
        self.out_conv=nn.Sequential(*[nn.Conv2d(in_channels=channels*len(shifts),out_channels=channels,kernel_size=1,stride=1),
                                      nn.BatchNorm2d(channels),
                                      nn.PReLU()])
        self.silu=torch.nn.SiLU()
    def forward(self,cen):
        outs=[]
        for index,shift in enumerate(self.shifts):
            outs.append(self.convs_list[index](cen))
        outs=torch.concatenate(outs,dim=1)
        outs=self.out_conv(outs)+cen
        outs=self.silu(outs)
        return outs