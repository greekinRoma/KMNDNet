from torch import nn
import torch
class mix_layer(nn.Module):
    def __init__(self,num_layers):
        super().__init__()
        self.paras = torch.nn.Parameter(torch.zeros([num_layers-1]).cuda())
        self.act=torch.sigmoid
    def forward(self,xins,share_x):
        outs=[]
        print(self.paras)
        for i,xin in enumerate(xins):
            outs.append(xin+share_x*self.act(self.paras[i]))
        return outs