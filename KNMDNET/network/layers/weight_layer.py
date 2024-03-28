from torch import nn
import torch
class weight_layer(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.para = torch.nn.Parameter(torch.zeros(1,in_channel,1,1).cuda()).cuda()
        self.act=torch.sigmoid
    def forward(self,x):
        kernel = (x.size()[2], x.size()[3])
        weight=self.act(self.para)
        print(self.para)
        out = weight.repeat(x.shape[0], 1, kernel[0], kernel[1])
        x=out*x
        return x