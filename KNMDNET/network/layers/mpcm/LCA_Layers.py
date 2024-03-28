#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from ..network_blocks import BaseConv, CSPLayer, DWConv
from ..layers.MLC_layer import MLC

class LCA_Layer(nn.Module):
    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_convs=torch.nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        for channels in in_channels:
            self.out_convs.append(Conv(in_channels=int(width*channels),
                                    out_channels=1,
                                       stride=1,
                                       ksize=1))
        self.mlc1=MLC(in_channels=int(width*in_channels[0]),out_channels=1,scale=8,i=1,shifts=[1,3,5,7])
        self.mlc0=MLC(in_channels=int(width*in_channels[1]),out_channels=1,scale=8,i=2,shifts=[1,3,5,7])
    def forward(self, features):
        [x2, x1, x0] = features
        mask_out0=x0
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16
        mask_out1=f_out0
        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        mask_out2 = self.C3_p3(f_out1)  # 512->256/8
        mask_outs=[mask_out2,mask_out1,mask_out0]
        outs=[]
        for k,mask_out in enumerate(mask_outs):
            outs.append(self.out_convs[k](mask_out))
        # for out in outs:
        #     print(out.shape)
        # mask1=self.mlc1(cen=x2,mask=torch.sigmoid(outs[1]))
        # mask0=self.mlc0(cen=x1, mask=torch.sigmoid(outs[2]))
        return features,outs