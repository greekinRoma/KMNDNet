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
class C3modul_0(nn.Module):
    def __init__(self, channels, add=True, shifts=[2, 4, 8, 16]):
        super().__init__()
        n = int(channels / 4)
        self.c1 = nn.Conv2d(channels,n,kernel_size=1,stride=1,bias=False)
        self.d1 = C3block(n, shifts[0])
        self.d2 = C3block(n, shifts[1])
        self.d3 = C3block(n, shifts[2])
        self.d4 = C3block(n, shifts[3])
        self.bn = nn.BatchNorm2d(channels)
        self.act= nn.PReLU(channels)
        self.add = add
    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)
        combine = torch.cat([d1, d2, d3, d4], 1)
        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        output=self.act(output)
        return output
class C3modul_1(nn.Module):
    def __init__(self, channels, add=True, shifts=[2, 4, 8, 16]):
        super().__init__()
        n = int(channels / 4)
        self.c1 = nn.Conv2d(channels,n,kernel_size=1,stride=1,bias=False)
        self.d1 = nn.Sequential(*[nn.Conv2d(in_channels=n,out_channels=n,kernel_size=3,dilation=shifts[0],padding=shifts[0]),
                                  nn.BatchNorm2d(n),
                                  nn.PReLU(n)])
        self.d2 = nn.Sequential(*[nn.Conv2d(in_channels=n,out_channels=n,kernel_size=3,dilation=shifts[1],padding=shifts[1]),
                                  nn.BatchNorm2d(n),
                                  nn.PReLU(n)])
        self.d3 = nn.Sequential(*[nn.Conv2d(in_channels=n,out_channels=n,kernel_size=3,dilation=shifts[2],padding=shifts[2]),
                                  nn.BatchNorm2d(n),
                                  nn.PReLU(n)])
        self.d4 = nn.Sequential(*[nn.Conv2d(in_channels=n,out_channels=n,kernel_size=3,dilation=shifts[3],padding=shifts[3]),
                                  nn.BatchNorm2d(n),
                                  nn.PReLU(n)])
        self.bn = nn.BatchNorm2d(channels)
        self.act= nn.PReLU(channels)
        self.add = add
    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)
        combine = torch.cat([d1, d2, d3, d4], 1)
        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        output=self.act(output)
        return output
class C3modul_2(nn.Module):
    def __init__(self, channels, add=True, shifts=[3, 5, 7, 9]):
        super().__init__()
        n = int(channels / 4)
        self.c1 = nn.Conv2d(channels,n,kernel_size=1,stride=1,bias=False)
        self.d1 = C3block(n, shifts[0])
        self.d2 = C3block(n, shifts[1])
        self.d3 = C3block(n, shifts[2])
        self.d4 = C3block(n, shifts[3])
        self.bn = nn.BatchNorm2d(channels)
        self.act= nn.PReLU(channels)
        self.add = add
    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)
        combine = torch.cat([d1, d2, d3, d4], 1)
        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        output=self.act(output)
        return output