import torch
from torch import nn
from ..network_blocks import BaseConv
import torch.functional as F
class calayer(nn.Module):
    def __init__(self,in_channels,width,height,scale):
        super().__init__()
        self.in_channels=in_channels
        self.width=width
        self.height=height
        hidden_channels=int(scale*in_channels)
        self.hidden_channels=hidden_channels
        self.concat_conv=BaseConv(in_channels=width+height,out_channels=width+height,ksize=1,stride=1)
        self.x_conv=nn.Sequential(*[nn.Conv2d(in_channels=in_channels,out_channels=hidden_channels,kernel_size=1,stride=1)])
        self.y_conv=nn.Sequential(*[nn.Conv2d(in_channels=in_channels,out_channels=hidden_channels,kernel_size=1,stride=1)])
        self.out_conv=torch.nn.Sequential(*[BaseConv(in_channels=hidden_channels,out_channels=1,stride=1,ksize=1),
                                            nn.Conv2d(in_channels=1,out_channels=1,stride=1,kernel_size=3,padding=1),
                                            nn.Sigmoid()])
    def forward(self,inp):
        x_inp=torch.mean(inp,dim=2,keepdim=True)
        y_inp=torch.mean(inp,dim=3,keepdim=True)
        x_inp=x_inp.permute(0,3,2,1)
        y_inp=y_inp.permute(0,2,3,1)
        xy_inp=torch.cat([x_inp,y_inp],dim=1)
        xy_inp=self.concat_conv(xy_inp)
        x_out,y_out=torch.split(xy_inp,[self.width,self.height],dim=1)
        x_out=(x_out+x_inp).permute(0,3,1,2)
        y_out=(y_out+x_inp).permute(0,3,2,1)
        x_out=self.x_conv(x_out)
        y_out=self.y_conv(y_out)
        mask=x_out*y_out
        mask=self.out_conv(mask)
        return mask