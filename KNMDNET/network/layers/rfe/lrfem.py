from torch import nn
import torch
from ...network_blocks import BaseConv
class RFPModule(nn.Module):
    def __init__(self,channels,shift):
        super().__init__()
        self.channels=channels
        self.shift=shift
        self.conv=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=2*shift+1,padding=shift,groups=channels,bias=False)
        self.out_conv=nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,bias=False))
    def forward(self,cen):
        out=self.conv(cen)
        out=self.out_conv(out)
        return out
class RFP(nn.Module):
    def __init__(self, in_channels, shifts=[2,3,4]):
        super().__init__()
        self.channels=in_channels
        self.conv1=torch.nn.Sequential(*[BaseConv(in_channels=in_channels, out_channels=in_channels // 2, ksize=1, stride=1, bias=False),
                                         RFPModule(channels=in_channels // 2, shift=shifts[0])])
        self.conv2=torch.nn.Sequential(*[BaseConv(in_channels=int(in_channels/2), out_channels=in_channels // 4, ksize=1, stride=1, bias=False),
                                         RFPModule(channels=in_channels // 4, shift=shifts[1])])
        self.conv3=torch.nn.Sequential(*[BaseConv(in_channels=int(in_channels *3/ 4), out_channels=in_channels // 4, ksize=1, stride=1, bias=False),
                                         RFPModule(channels=in_channels // 4, shift=shifts[2])])
        self.out_conv=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,bias=False)
        self.act=nn.Sequential(*[nn.BatchNorm2d(in_channels),
                                 nn.SiLU()])
    def forward(self,input):
        out1=self.conv1(input)
        out2=self.conv2(out1)
        out2=torch.concat([out1,out2],dim=1)
        out3=self.conv3(out2)
        out=torch.concat([out2,out3],dim=1)
        out=self.out_conv(out)
        out=self.act(out)
        return out
class RFP_1(nn.Module):
    def __init__(self,channels,shifts:list=[1,2,4,8]):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.d1=RFPModule(channels=channels//4,shift=shifts[0])
        self.d2=RFPModule(channels=channels//4,shift=shifts[1])
        self.d4=RFPModule(channels=channels//4,shift=shifts[2])
        self.d8=RFPModule(channels=channels//4,shift=shifts[3])
        self.outconv=nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
    def forward(self,inp):
        out0=self.c1(inp)
        out1=self.d1(out0)+out0
        out2=self.d2(out1)+out1
        out3=self.d4(out2)+out2
        out4=self.d8(out3)+out3
        out=torch.concat([out1,out2,out3,out4],dim=1)
        out=self.outconv(out)
        return out
class RFP_2(nn.Module):
    def __init__(self,channels,shifts:list=[1,2,4,8]):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.c2=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.c3=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.c4 = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=1, stride=1)
        self.d1=RFPModule(channels=channels//4,shift=shifts[0])
        self.d2=RFPModule(channels=channels//4,shift=shifts[1])
        self.d4=RFPModule(channels=channels//4,shift=shifts[2])
        self.d8=RFPModule(channels=channels//4,shift=shifts[3])
        self.outconv=nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
    def forward(self,inp):
        out0=self.c1(inp)
        out1=self.c2(inp)
        out2=self.c3(inp)
        out3=self.c4(inp)
        out0=self.d1(out0)+out0
        out1=self.d2(out1)+out1
        out2=self.d4(out2)+out2
        out3=self.d8(out3)+out3
        out=torch.concat([out0,out1,out2,out3],dim=1)
        out=self.outconv(out)
        return out
class RFP_3(nn.Module):
    def __init__(self,channels,shifts:list=[1,3,5,9]):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.c2=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.c3=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.c4 = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=1, stride=1)
        self.d1=RFPModule(channels=channels//4,shift=shifts[0])
        self.d2=RFPModule(channels=channels//4,shift=shifts[1])
        self.d4=RFPModule(channels=channels//4,shift=shifts[2])
        self.d8=RFPModule(channels=channels//4,shift=shifts[3])
    def forward(self,inp):
        out0=self.c1(inp)
        out1=self.c2(inp)
        out2=self.c3(inp)
        out3=self.c4(inp)
        out0=self.d1(out0)+out0
        out1=self.d2(out1)+out1
        out2=self.d4(out2)+out2
        out3=self.d8(out3)+out3
        out=torch.concat([out0,out1,out2,out3],dim=1)
        return out
class RFP_4(nn.Module):
    def __init__(self,channels,shifts:list=[1,3,5,9]):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.c2=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.c3=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.c4 = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=1, stride=1)
        self.d1=RFPModule(channels=channels//4,shift=shifts[0])
        self.d2=RFPModule(channels=channels//4,shift=shifts[1])
        self.d4=RFPModule(channels=channels//4,shift=shifts[2])
        self.d8=RFPModule(channels=channels//4,shift=shifts[3])
        self.resconv=nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=1,padding=1),)
        self.outconv=nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=1,padding=1))
        self.act=nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
    def forward(self,inp):
        out0=self.c1(inp)
        out1=self.c2(inp)
        out2=self.c3(inp)
        out3=self.c4(inp)
        out0=self.d1(out0)+out0
        out1=self.d2(out1)+out1
        out2=self.d4(out2)+out2
        out3=self.d8(out3)+out3
        out=torch.concat([out0,out1,out2,out3],dim=1)
        out=self.resconv(inp)+self.outconv(out)
        out=self.act(out)
        return out
class RFP_5(nn.Module):
    def __init__(self,channels,shifts:list=[3,5,7,9]):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.c2=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.c3=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.c4 = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=1, stride=1)
        self.d1=RFPModule(channels=channels//4,shift=shifts[0])
        self.d2=RFPModule(channels=channels//4,shift=shifts[1])
        self.d4=RFPModule(channels=channels//4,shift=shifts[2])
        self.d8=RFPModule(channels=channels//4,shift=shifts[3])
        self.inp_conv=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=1,padding=1)
        self.outconv=nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,bias=False),
            nn.Sigmoid()
        )
    def forward(self,inp):
        out0=self.c1(inp)
        out1=self.c2(inp)
        out2=self.c3(inp)
        out3=self.c4(inp)
        out0=self.d1(out0)+out0
        out1=self.d2(out1)+out1
        out2=self.d4(out2)+out2
        out3=self.d8(out3)+out3
        out=torch.concat([out0,out1,out2,out3],dim=1)
        out=self.outconv(out)*self.outconv(inp)
        return out