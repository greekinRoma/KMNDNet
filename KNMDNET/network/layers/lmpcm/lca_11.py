import torch
from torch import nn
import numpy as np
from setting.read_setting import config as cfg
import math
class ExpansionContrastModule(nn.Module):
    def __init__(self,in_channels,shifts=[1,3,5,7]):
        super().__init__()
        self.convs_list=nn.ModuleList()
        self.shifts=shifts
        delta1=np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, -1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, -1], [0, 1, 0], [0, 0, 0]],[[0, 0, 0], [0, 1, -1], [0, 0, 0]]])
        delta1=delta1.reshape(4,1,3,3)
        delta2=delta1[:,:,::-1,::-1].copy()
        delta=np.concatenate([delta1,delta2],axis=0)
        w1,w2,w3,w4,w5,w6,w7,w8=np.array_split(delta,8)
        self.in_channels = in_channels//4
        self.sum_kernel=torch.ones([self.in_channels,8,1,1])
        if cfg.use_cuda:
            self.kernel1 = torch.Tensor(w1).cuda()
            self.kernel2 = torch.Tensor(w2).cuda()
            self.kernel3 = torch.Tensor(w3).cuda()
            self.kernel4 = torch.Tensor(w4).cuda()
            self.kernel5 = torch.Tensor(w5).cuda()
            self.kernel6 = torch.Tensor(w6).cuda()
            self.kernel7 = torch.Tensor(w7).cuda()
            self.kernel8 = torch.Tensor(w8).cuda()
            self.sum_kernel=self.sum_kernel.cuda()
        else:
            self.kernel1 = torch.Tensor(w1)
            self.kernel2 = torch.Tensor(w2)
            self.kernel3 = torch.Tensor(w3)
            self.kernel4 = torch.Tensor(w4)
            self.kernel5 = torch.Tensor(w5)
            self.kernel6 = torch.Tensor(w6)
            self.kernel7 = torch.Tensor(w7)
            self.kernel8 = torch.Tensor(w8)
        self.kernel1 = self.kernel1.repeat(self.in_channels,1,1,1).contiguous()
        self.kernel2 = self.kernel2.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel3 = self.kernel3.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel4 = self.kernel4.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel5 = self.kernel5.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel6 = self.kernel6.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel7 = self.kernel7.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel8 = self.kernel8.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.in_conv=nn.Conv2d(in_channels=in_channels,out_channels=self.in_channels,kernel_size=1,stride=1)
        self.out_conv=nn.Conv2d(in_channels=self.in_channels,out_channels=1,kernel_size=1,stride=1)
        self.layer1=nn.Conv2d(in_channels=self.in_channels*8,out_channels=self.in_channels*8,kernel_size=1,stride=1,groups=self.in_channels)
        self.layer2=nn.Conv2d(in_channels=self.in_channels*8,out_channels=self.in_channels,kernel_size=1,stride=1,groups=self.in_channels)
        self.layer3 = nn.Conv2d(in_channels=self.in_channels * 4, out_channels=self.in_channels, kernel_size=1,
                                stride=1, groups=self.in_channels)
        for shift in shifts:
            self.convs_list.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=shift,stride=1,padding=shift//2,groups=self.in_channels))
    def circ_shift(self,cen,index,shift):
        b,c,w,h=cen.shape
        cen=self.convs_list[index](cen)
        delta1 = torch.nn.functional.conv2d(weight=self.kernel1, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        delta2 = torch.nn.functional.conv2d(weight=self.kernel2, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        delta3 = torch.nn.functional.conv2d(weight=self.kernel3, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        delta4 = torch.nn.functional.conv2d(weight=self.kernel4, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        delta5 = torch.nn.functional.conv2d(weight=self.kernel5, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        delta6 = torch.nn.functional.conv2d(weight=self.kernel6, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        delta7 = torch.nn.functional.conv2d(weight=self.kernel7, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        delta8 = torch.nn.functional.conv2d(weight=self.kernel8, stride=1, padding="same", dilation=shift, input=cen,
                                          groups=self.in_channels)
        inp1 = torch.stack([delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8], dim=2).view(b, -1, w, h).contiguous()
        inp2 = torch.stack([delta2, delta3, delta4, delta5, delta6, delta7, delta8, delta1], dim=2).view(b, -1, w, h).contiguous()
        inp3 = torch.stack([delta3, delta4, delta5, delta6, delta7, delta8, delta1, delta2], dim=2).view(b, -1, w, h).contiguous()
        inp4 = torch.stack([delta4, delta5, delta6, delta7, delta8, delta1, delta2, delta3], dim=2).view(b, -1, w, h).contiguous()
        inp5 = torch.stack([delta5, delta6, delta7, delta8, delta1, delta2, delta3, delta4], dim=2).view(b, -1, w, h).contiguous()
        inp6 = torch.stack([delta6, delta7, delta8, delta1, delta2, delta3, delta4, delta5], dim=2).view(b, -1, w, h).contiguous()
        inp7 = torch.stack([delta8, delta8, delta1, delta2, delta3, delta4, delta5, delta6], dim=2).view(b, -1, w, h).contiguous()
        inp8 = torch.stack([delta7, delta1, delta2, delta3, delta4, delta5, delta6, delta7], dim=2).view(b, -1, w, h).contiguous()
        tmp1 = self.layer1(inp1) * self.layer1(inp5)
        tmp2 = self.layer1(inp2) * self.layer1(inp6)
        tmp3 = self.layer1(inp3) * self.layer1(inp7)
        tmp4 = self.layer1(inp4) * self.layer1(inp8)
        tmp1 = self.layer2(tmp1)
        tmp2 = self.layer2(tmp2)
        tmp3 = self.layer2(tmp3)
        tmp4 = self.layer2(tmp4)
        outs = torch.stack([tmp1,tmp2,tmp3,tmp4],dim=2)
        outs = torch.sort(outs, dim=2).values
        outs = outs.view(b, -1, w, h).contiguous()
        outs = self.layer3(outs)
        return outs
    def spatial_attention(self,cen):
        outs=[]
        cen=self.in_conv(cen)
        for index,shift in enumerate(self.shifts):
            outs.append(self.circ_shift(cen,index,shift))
        outs=torch.stack(outs,dim=-1)
        outs=torch.max(outs,dim=-1).values+torch.mean(outs,dim=-1)
        outs=self.out_conv(outs)
        return outs
    def forward(self,cen,mas):
        outs=self.spatial_attention(cen)
        # import cv2
        # import numpy as np
        # outs_0=np.array(torch.sigmoid(outs).detach().cpu())
        # print(np.max(outs_0))
        # print(np.min(outs_0))
        # outs_0=outs_0[0][0]
        # cv2.imshow("outcome",outs_0)
        # cv2.waitKey(0)
        return outs