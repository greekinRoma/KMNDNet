from torch import nn
import torch
from network.backbones.ssd_backbone import SSDBackbone
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}
class SSDHead(nn.Module):
    def __init__(self,backbone:SSDBackbone,size:int=300,num_classes:int=1):
        super().__init__()
        self.num_classes=num_classes
        self.loc_layers,self.conf_layers=self.multibox(backbone,cfg=mbox[str(size)],num_classes=self.num_classes)
    def multibox(self,backbone:SSDBackbone, cfg, num_classes):
        loc_layers = []
        conf_layers = []
        vgg_source = [21, -2]
        for k, v in enumerate(vgg_source):
            loc_layers += [nn.Conv2d(backbone.vgg[v].out_channels,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(backbone.vgg[v].out_channels,
                                      cfg[k] * num_classes, kernel_size=3, padding=1)]
        for k, v in enumerate(backbone.extras[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]
        return loc_layers, conf_layers
    def forward(self,sources):
        loc=list()
        conf=list()
        for (x,l,c) in zip(sources,self.loc_layers,self.conf_layers):
            loc.append(l(x).permute(0,2,3,1).contiguous())
            conf.append(c(x).permute(0,2,3,1).contiguous())
        loc=torch.cat([o.view(o.size(0),-1) for o in loc],dim=1)
        conf=torch.cat([o.view(o.size(0),-1) for o in conf],dim=1)
        output=torch.concat([loc.view(loc.size(0),-1,4),
                             conf.view(conf.size(0),-1,self.num_classes)],dim=-1)
        return output
