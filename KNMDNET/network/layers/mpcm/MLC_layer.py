import torch
from torch import nn
import numpy as np
from ..network_blocks import BaseConv
from setting.read_setting import config as cfg
class MLC(nn.Module):
    def __init__(self,in_channels,out_channels,scale,i,shifts=[1,3]):
        super().__init__()
        self.convs_list=nn.ModuleList()
        self.out_convs_list=nn.ModuleList()
        self.avepools_list=nn.ModuleList()
        self.layer1_list=nn.ModuleList()
        self.layer2_list=nn.ModuleList()
        self.scale_list=nn.ModuleList()
        self.down_list=nn.ModuleList()
        self.shifts=shifts
        self.scale=scale
        w1=np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, -1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, -1], [0, 1, 0], [0, 0, 0]],[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]])
        w1=w1.reshape(4,1,3,3)
        w2=w1[:,:,::-1,::-1].copy()
        if cfg.use_cuda:
            self.kernel1=torch.Tensor(w1).cuda()
            self.kernel2=torch.Tensor(w2).cuda()
            self.params =torch.nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding=1).cuda().bias
        else:
            self.kernel1 = torch.Tensor(w1)
            self.kernel2 = torch.Tensor(w2)
            self.params =torch.nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding=1).bias
        for shift in shifts:
            self.scale_list.append(nn.Conv2d(in_channels=4,out_channels=16,kernel_size=1,stride=1,bias=False))
            # self.down_list.append(nn.Sequential(*[BaseConv(in_channels=in_channels,out_channels=in_channels,ksize=3,stride=1)])
            self.convs_list.append(nn.Conv2d(in_channels=1,out_channels=1,kernel_size=shift,stride=1,padding=shift//2))
            self.avepools_list.append(nn.AvgPool2d(kernel_size=shift,stride=1,padding=shift//2))
        hidden_channel = 8 * 4 ** i
        self.down_conv=nn.Sequential(*[BaseConv(in_channels=in_channels,out_channels=in_channels,stride=1,ksize=3),
                                       nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=1,stride=1)])
        self.out_conv=nn.Sequential(*[BaseConv(in_channels=in_channels,out_channels=in_channels//2,ksize=1,stride=1),
                                      nn.Conv2d(in_channels=in_channels//2,out_channels=in_channels,kernel_size=1,stride=1),
                                      torch.nn.Sigmoid()])
        self.act=torch.nn.Sigmoid()
    def circ_shift(self,cen,index,shift):
        # cen=torch.mean(cen,dim=1,keepdim=True)
        # print(cen.shape)
        # print(torch.max(cen))
        # print(torch.min(cen))
        out1=torch.nn.functional.conv2d(weight=self.kernel1,stride=1,padding="same",dilation=shift,input=cen)
        out2=torch.nn.functional.conv2d(weight=self.kernel2,stride=1,padding="same",dilation=shift,input=cen)
        # out1=torch.relu(out1)
        # out2=torch.relu(out2)
        out=out1*out2
        out=torch.min(out,dim=1,keepdim=True)[0]
        # print(torch.max(out))
        # print(torch.min(out))
        # import cv2
        # for o in out:
        #     o=torch.sigmoid(o)
        #     o=o.permute(1,2,0)
        #     o=np.array(o.detach().cpu())
        #     cv2.imshow("outcome_{}".format(o.shape[0]),o)
        #     cv2.waitKey(0)
        return out
    def spatial_attention(self,cen):
        outs=[]
        cen=self.down_conv(cen)
        for index,shift in enumerate(self.shifts):
            outs.append(self.circ_shift(cen,index,shift))
        outs=torch.stack(outs,dim=-1)
        outs=(torch.max(outs,dim=-1,keepdim=False)[0]+torch.mean(outs,dim=-1,keepdim=False))/2
        outs=torch.relu(outs)
        out=torch.sigmoid(outs)
        return out
    def forward(self,cen):
        local_mask=self.spatial_attention(cen)
        out_mask=self.out_conv(local_mask*cen)
        return out_mask*cen+cen