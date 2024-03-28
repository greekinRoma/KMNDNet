from torch import nn
import torch
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
class ESP_Module(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.d1=nn.Sequential(*[nn.Conv2d(in_channels=channels//4,out_channels=channels//4,kernel_size=3,dilation=1,padding=1,bias=False),
                                nn.BatchNorm2d(channels//4),
                                nn.PReLU(channels//4)])
        self.d2 = nn.Sequential(
            *[nn.Conv2d(in_channels=channels//4, out_channels=channels//4, kernel_size=3, dilation=2, padding=2,bias=False),
              nn.BatchNorm2d(channels//4),
              nn.PReLU(channels//4)])
        self.d4 = nn.Sequential(
            *[nn.Conv2d(in_channels=channels // 4, out_channels=channels // 4, kernel_size=3, dilation=4, padding=4,
                        bias=False),
              nn.BatchNorm2d(channels // 4),
              nn.PReLU(channels // 4)])
        self.d8=nn.Sequential(
            *[nn.Conv2d(in_channels=channels // 4, out_channels=channels // 4, kernel_size=3, dilation=8, padding=8,
                        bias=False),
              nn.BatchNorm2d(channels // 4),
              nn.PReLU(channels // 4)])
        self.bn = nn.BatchNorm2d(channels)
        self.act=nn.PReLU()
    def forward(self,inp):
        out0=self.c1(inp)
        out1=self.d1(out0)
        out2=self.d2(out0)+out1
        out3=self.d4(out0)+out2
        out4=self.d8(out0)+out3
        out=torch.concat([out1,out2,out3,out4],dim=1)
        out=self.bn(out+inp)
        out=self.act(out)
        return out
class ESP_Module_1(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=channels,out_channels=channels//4,kernel_size=1,stride=1)
        self.d1=C3block(channels=channels//4,d=1)
        self.d2 = C3block(channels=channels//4,d=2)
        self.d4 = C3block(channels=channels//4,d=4)
        self.d8=C3block(channels=channels//4,d=8)
        self.bn = nn.BatchNorm2d(channels)
        self.act=nn.PReLU()
    def forward(self,inp):
        out0=self.c1(inp)
        out1=self.d1(out0)
        out2=self.d2(out0)+out1
        out3=self.d4(out0)+out2
        out4=self.d8(out0)+out3
        out=torch.concat([out1,out2,out3,out4],dim=1)
        out=self.bn(out+inp)
        out=self.act(out)
        return out