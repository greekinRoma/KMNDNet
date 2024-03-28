import torch
from torch import nn
import numpy as np
from ...network_blocks import BaseConv
from setting.read_setting import config as cfg
class ExpansionContrastModule(nn.Module):
    def __init__(self,in_channels,shifts=[1,3]):
        super().__init__()
        self.convs_list=nn.ModuleList()
        self.out_convs_list=nn.ModuleList()
        self.avepools_list=nn.ModuleList()
        self.layer1_list=nn.ModuleList()
        self.layer2_list=nn.ModuleList()
        self.scale_list=nn.ModuleList()
        self.down_list=nn.ModuleList()
        self.shifts=shifts
        w1=np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, -1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, -1], [0, 1, 0], [0, 0, 0]],[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]])
        w1=w1.reshape(4,1,3,3)
        w2=w1[:,:,::-1,::-1].copy()
        if cfg.use_cuda:
            self.kernel1=torch.Tensor(w1).cuda()
            self.kernel2=torch.Tensor(w2).cuda()
            self.scales=nn.Parameter(torch.zeros(4).cuda())
            self.params =torch.nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding=1).cuda().bias
        else:
            self.kernel1 = torch.Tensor(w1)
            self.kernel2 = torch.Tensor(w2)
            self.params =torch.nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding=1).bias
            self.scales=torch.nn.Parameter(torch.zeros(4))
        self.kernel1=self.kernel1.repeat(in_channels//4,1,1,1).contiguous()
        self.kernel2=self.kernel2.repeat(in_channels//4,1,1,1).contiguous()
        self.in_channels=in_channels//4
        self.act=torch.nn.Sigmoid()
        self.in_conv=nn.Conv2d(in_channels=in_channels,out_channels=self.in_channels,kernel_size=1,stride=1)
        self.out_conv=nn.Sequential(
            BaseConv(in_channels=self.in_channels,out_channels=self.in_channels,ksize=1,stride=1),
            nn.Conv2d(in_channels=self.in_channels,out_channels=1,kernel_size=1,stride=1)
        )
    def circ_shift(self,cen,index,shift):
        # cen=torch.mean(cen,dim=1,keepdim=True)
        # print(cen.shape)
        # print(torch.max(cen))
        # print(torch.min(cen))
        batch_size,num_channels,height,widht=cen.shape
        out1=torch.nn.functional.conv2d(weight=self.kernel1,stride=1,padding="same",dilation=shift,input=cen,groups=self.in_channels)
        out2=torch.nn.functional.conv2d(weight=self.kernel2,stride=1,padding="same",dilation=shift,input=cen,groups=self.in_channels)
        # out1=torch.relu(out1)
        # out2=torch.relu(out2)
        out=out1*out2
        out=out.view(batch_size,num_channels,4,height,widht).contiguous()
        out=torch.min(out,dim=2).values
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
        cen=self.in_conv(cen)
        for index,shift in enumerate(self.shifts):
            outs.append(self.circ_shift(cen,index,shift))
        outs=torch.stack(outs,dim=-1)
        outs=torch.max(outs,dim=-1,keepdim=False).values+torch.mean(outs,dim=-1,keepdim=False)
        outs=torch.relu(outs)
        outs=self.out_conv(outs)
        out=torch.sigmoid(outs)
        return out
    def forward(self,cen,mas):
        out_mask=self.spatial_attention(cen)
        # print(self.scales)
        scales=torch.softmax(self.scales,dim=-1)
        # mask=(out_mask * mas.sigmoid() * scales[0] + mas.sigmoid() * scales[1] + scales[2] * out_mask + scales[3])
        # import cv2
        # import numpy as np
        # mask=np.array(mask.detach().cpu())[0,0]
        # print(mask.shape)
        # cv2.imshow("outcome",mask)
        # cv2.waitKey(0)
        # print(scales)
        return cen*(out_mask*mas.sigmoid()*scales[0]+mas.sigmoid()*scales[1]+scales[2]*out_mask+scales[3])