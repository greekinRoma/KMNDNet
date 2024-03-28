import torch
from torch import nn
from network.network_blocks import BaseConv
class RFEM(nn.Module):
    def __init__(self, channels, shifts=[2, 4, 8]):
        super().__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=3, stride=1,
                                     dilation=shifts[0], padding=shifts[0])
        self.conv2 = torch.nn.Sequential(
            *[BaseConv(in_channels=int(channels * 3 / 2), out_channels=channels // 4, ksize=1, stride=1),
              torch.nn.Conv2d(in_channels=channels // 4, out_channels=channels // 4, kernel_size=3, stride=1,
                              dilation=shifts[1], padding=shifts[1])])
        self.conv3 = torch.nn.Sequential(
            *[BaseConv(in_channels=int(channels * 7 / 4), out_channels=channels // 4, ksize=1, stride=1),
              torch.nn.Conv2d(in_channels=channels // 4, out_channels=channels // 4, kernel_size=3, stride=1,
                              dilation=shifts[2], padding=shifts[2])])
        self.out_conv =BaseConv(in_channels=channels*2, out_channels=channels, ksize=1, stride=1)
    def forward(self,input):
        out1=self.conv1(input)
        out1=torch.concat([input,out1],dim=1)
        out2=self.conv2(out1)
        out2=torch.concat([out1,out2],dim=1)
        out3=self.conv3(out2)
        out=torch.concat([out2,out3],dim=1)
        out=self.out_conv(out)
        return out
class RFEM_1(nn.Module):
    def __init__(self,channels,shifts=[2,4,8]):
        super().__init__()
        self.channels=channels
        self.conv1=torch.nn.Conv2d(in_channels=channels,out_channels=channels//2,kernel_size=3,stride=1,dilation=shifts[0],padding=shifts[0])
        self.conv2=torch.nn.Sequential(*[BaseConv(in_channels=int(channels*3/2),out_channels=channels//4,ksize=1,stride=1),
                                         torch.nn.Conv2d(in_channels=channels//4,out_channels=channels//4,kernel_size=3,stride=1,dilation=shifts[1],padding=shifts[1])])
        self.conv3=torch.nn.Sequential(*[BaseConv(in_channels=int(channels*7/4),out_channels=channels//4,ksize=1,stride=1),
                                         torch.nn.Conv2d(in_channels=channels//4,out_channels=channels//4,kernel_size=3,stride=1,dilation=shifts[2],padding=shifts[2])])
        self.out_conv=nn.Conv2d(in_channels=channels*2,out_channels=channels,kernel_size=1,stride=1)
        self.act=nn.Sequential(*[nn.BatchNorm2d(channels),
                                 nn.SiLU()])
    def forward(self,input):
        out1=self.conv1(input)
        out1=torch.concat([input,out1],dim=1)
        out2=self.conv2(out1)
        out2=torch.concat([out1,out2],dim=1)
        out3=self.conv3(out2)
        out=torch.concat([out2,out3],dim=1)
        out=self.out_conv(out)
        out=self.act(input+out)
        return out
class RFEM_2(nn.Module):
    def __init__(self,channels,shifts=[3,5,9]):
        super().__init__()
        self.channels=channels
        self.conv1=torch.nn.Conv2d(in_channels=channels,out_channels=channels//2,kernel_size=3,stride=1,dilation=shifts[0],padding=shifts[0],bias=False)
        self.conv2=torch.nn.Sequential(*[BaseConv(in_channels=int(channels/2),out_channels=channels//4,ksize=1,stride=1,bias=False),
                                         torch.nn.Conv2d(in_channels=channels//4,out_channels=channels//4,bias=False,kernel_size=3,stride=1,dilation=shifts[1],padding=shifts[1])])
        self.conv3=torch.nn.Sequential(*[BaseConv(in_channels=int(channels*3/4),out_channels=channels//4,ksize=1,stride=1,bias=False),
                                         torch.nn.Conv2d(in_channels=channels//4,out_channels=channels//4,kernel_size=3,stride=1,dilation=shifts[2],padding=shifts[2],bias=False)])
        self.out_conv=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,bias=False)
        self.act=nn.Sequential(*[nn.BatchNorm2d(channels),
                                 nn.SiLU()])
    def forward(self,input):
        out1=self.conv1(input)
        out2=self.conv2(out1)
        out2=torch.concat([out1,out2],dim=1)
        out3=self.conv3(out2)
        out=torch.concat([out2,out3],dim=1)
        out=self.out_conv(out)
        out=self.act(input+out)
        return out
class RFEM_3(nn.Module):
    def __init__(self,channels,shifts=[1,3,5,9]):
        super().__init__()
        self.channels=channels
        self.conv1=torch.nn.Sequential(*[BaseConv(in_channels=channels,out_channels=channels//4,ksize=1,stride=1,bias=False),
                                         torch.nn.Conv2d(in_channels=channels//4,out_channels=channels//4,kernel_size=3,stride=1,dilation=shifts[0],padding=shifts[0],bias=False)])
        self.conv2=torch.nn.Sequential(*[BaseConv(in_channels=channels//4,out_channels=channels//4,ksize=1,stride=1,bias=False),
                                         torch.nn.Conv2d(in_channels=channels//4,out_channels=channels//4,kernel_size=3,stride=1,dilation=shifts[1],padding=shifts[1],bias=False)])
        self.conv3=torch.nn.Sequential(*[BaseConv(in_channels=channels//2,out_channels=channels//4,ksize=1,stride=1,bias=False),
                                         torch.nn.Conv2d(in_channels=channels//4,out_channels=channels//4,kernel_size=3,stride=1,dilation=shifts[2],padding=shifts[2],bias=False)])
        self.conv4 = torch.nn.Sequential(*[BaseConv(in_channels=channels*3// 4, out_channels=channels // 4, ksize=1, stride=1, bias=False),
              torch.nn.Conv2d(in_channels=channels // 4, out_channels=channels // 4, kernel_size=3, stride=1,dilation=shifts[3], padding=shifts[3], bias=False)])
        self.out_conv=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,bias=False)
        self.act=nn.Sequential(*[nn.BatchNorm2d(channels),
                                 nn.SiLU()])
    def forward(self,input):
        out1=self.conv1(input)
        out2=self.conv2(out1)
        out2=torch.concat([out1,out2],dim=1)
        out3=self.conv3(out2)
        out3=torch.concat([out2,out3],dim=1)
        out4=self.conv4(out3)
        out=torch.concat([out3,out4],dim=1)
        out=self.out_conv(out)
        out=self.act(input+out)
        return out
class RFEM_4(nn.Module):
    def __init__(self,channels,shifts=[3,5,7]):
        super().__init__()
        self.channels=channels
        self.conv1=torch.nn.Conv2d(in_channels=channels,out_channels=channels//2,kernel_size=3,stride=1,dilation=shifts[0],padding=shifts[0],bias=False)
        self.conv2=torch.nn.Sequential(*[BaseConv(in_channels=int(channels/2),out_channels=channels//4,ksize=1,stride=1,bias=False),
                                         torch.nn.Conv2d(in_channels=channels//4,out_channels=channels//4,bias=False,kernel_size=3,stride=1,dilation=shifts[1],padding=shifts[1])])
        self.conv3=torch.nn.Sequential(*[BaseConv(in_channels=int(channels*3/4),out_channels=channels//4,ksize=1,stride=1,bias=False),
                                         torch.nn.Conv2d(in_channels=channels//4,out_channels=channels//4,kernel_size=3,stride=1,dilation=shifts[2],padding=shifts[2],bias=False)])
        self.out_conv=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,bias=False)
        self.act=nn.Sequential(*[nn.BatchNorm2d(channels),
                                 nn.SiLU()])
    def forward(self,input):
        out1=self.conv1(input)
        out2=self.conv2(out1)
        out2=torch.concat([out1,out2],dim=1)
        out3=self.conv3(out2)
        out=torch.concat([out2,out3],dim=1)
        out=self.out_conv(out)
        out=self.act(input+out)
        return out
class C3block_1(nn.Module):
    def __init__(self,channels,d):
        super().__init__()
        padding = d
        if d == 1:
            self.conv = nn.Conv2d(channels, channels, 1, stride=1, padding=1, bias=False,dilation=d)
        else:
            combine_kernel = 2 * d - 1
            self.conv = nn.Sequential(*[nn.Conv2d(channels, channels, kernel_size=(combine_kernel, 1), stride=1, padding=(padding - 1, 0),groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=(1, combine_kernel), stride=1, padding=(0, padding - 1),groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)])
    def forward(self,input):
        output=self.conv(input)
        return output
class RFEM_5(nn.Module):
    def __init__(self,channels,shifts=[1,5,9]):
        super().__init__()
        self.channels=channels
        self.conv1=torch.nn.Conv2d(in_channels=channels,out_channels=channels//2,kernel_size=3,stride=1,dilation=shifts[0],padding=shifts[0],bias=False)
        self.conv2=torch.nn.Sequential(*[BaseConv(in_channels=int(channels/2),out_channels=channels//4,ksize=1,stride=1,bias=False),
                                         torch.nn.Conv2d(in_channels=channels//4,out_channels=channels//4,bias=False,kernel_size=3,stride=1,dilation=shifts[1],padding=shifts[1])])
        self.conv3=torch.nn.Sequential(*[BaseConv(in_channels=int(channels*3/4),out_channels=channels//4,ksize=1,stride=1,bias=False),
                                         torch.nn.Conv2d(in_channels=channels//4,out_channels=channels//4,kernel_size=3,stride=1,dilation=shifts[2],padding=shifts[2],bias=False)])
        self.out_conv=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,bias=False)
        self.act=nn.Sequential(*[nn.BatchNorm2d(channels),
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
class RFEM_6(nn.Module):
    def __init__(self, channels, shifts=[3, 5, 9]):
        super().__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=3, stride=1,
                                     dilation=shifts[0], padding=shifts[0])
        self.conv2 = torch.nn.Sequential(
            *[BaseConv(in_channels=int(channels * 3 / 2), out_channels=channels // 4, ksize=1, stride=1),
              torch.nn.Conv2d(in_channels=channels // 4, out_channels=channels // 4, kernel_size=3, stride=1,
                              dilation=shifts[1], padding=shifts[1])])
        self.conv3 = torch.nn.Sequential(
            *[BaseConv(in_channels=int(channels * 7 / 4), out_channels=channels // 4, ksize=1, stride=1),
              torch.nn.Conv2d(in_channels=channels // 4, out_channels=channels // 4, kernel_size=3, stride=1,
                              dilation=shifts[2], padding=shifts[2])])
        self.out_conv =BaseConv(in_channels=channels*2, out_channels=channels, ksize=1, stride=1)
    def forward(self,input):
        out1=self.conv1(input)
        out1=torch.concat([input,out1],dim=1)
        out2=self.conv2(out1)
        out2=torch.concat([out1,out2],dim=1)
        out3=self.conv3(out2)
        out=torch.concat([out2,out3],dim=1)
        return out
class RFEM_7(nn.Module):
    def __init__(self,channels,shifts=[3,5,9]):
        super().__init__()
        self.channels=channels
        self.conv1=torch.nn.Conv2d(in_channels=channels,out_channels=channels//2,kernel_size=3,stride=1,dilation=shifts[0],padding=shifts[0],bias=False)
        self.conv2=torch.nn.Sequential(*[BaseConv(in_channels=int(channels/2),out_channels=channels//4,ksize=1,stride=1,bias=False),
                                         torch.nn.Conv2d(in_channels=channels//4,out_channels=channels//4,bias=False,kernel_size=3,stride=1,dilation=shifts[1],padding=shifts[1])])
        self.conv3=torch.nn.Sequential(*[BaseConv(in_channels=int(channels*3/4),out_channels=channels//4,ksize=1,stride=1,bias=False),
                                         torch.nn.Conv2d(in_channels=channels//4,out_channels=channels//4,kernel_size=3,stride=1,dilation=shifts[2],padding=shifts[2],bias=False)])
        self.out_conv=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,bias=False)
        self.act=nn.Sigmoid()
    def forward(self,input):
        out1=self.conv1(input)
        out2=self.conv2(out1)
        out2=torch.concat([out1,out2],dim=1)
        out3=self.conv3(out2)
        out=torch.concat([out2,out3],dim=1)
        out=self.out_conv(out)
        out=self.act(input)
        return out