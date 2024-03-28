from torch import nn
from network.layers.SSDModule import *
import torch
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}

class SSDBackbone(nn.Module):
    def __init__(self,size,num_classes:int=1):
        super().__init__()
        self.num_classes=num_classes
        self.vgg=nn.ModuleList(self.get_vgg(cfg=base[str(size)],i=3))
        self.extras=nn.ModuleList(self.add_extras(cfg=mbox[str(size)],i=1024))
        self.L2Norm=L2Norm(512,20)
    def add_extras(self,cfg, i, batch_norm=False):
        # Extra layers added to VGG for feature scaling
        layers = []
        in_channels = i
        flag = False
        for k, v in enumerate(cfg):
            if in_channels != 'S':
                if v == 'S':
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                         kernel_size=(1, 3)[flag], stride=2, padding=1)]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                flag = not flag
            in_channels = v
        return layers
    def get_vgg(self,cfg, i, batch_norm=False):
        layers = []
        in_channels = i
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        return layers
    def forward(self,x):
        sources=list()
        for k in range(23):
            x=self.vgg[k](x)
        s=self.L2Norm(x)
        sources.append(s)
        for k,v in enumerate(self.extras):
            x=nn.functional.relu(v(x),inplace=True)
            if k%2==1:
                sources.append(x)
        return sources