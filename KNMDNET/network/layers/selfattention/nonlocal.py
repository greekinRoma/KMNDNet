import torch
from torch import nn
import torch.nn.functional as F
class Nl(nn.Module):
    def __init__(self,in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(Nl, self).__init__()
        # self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.f_key = torch.nn.Sequential(*[nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,kernel_size=1, stride=1, padding=0)])
        self.f_query = self.f_key
        self.f_value =torch.nn.Sequential(*[nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,kernel_size=1, stride=1, padding=0)])
        self.W = torch.nn.Sequential(*[nn.Conv2d(in_channels=self.key_channels, out_channels=self.out_channels,kernel_size=1, stride=1, padding=0)])

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        print(query.shape)
        print(key.shape)
        sim_map = torch.matmul(query, key)
        # sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        return context