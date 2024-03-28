import torch
import torch.nn as nn
from ..network_blocks import BaseConv, DWConv
from ..layers.rfe.lrfem import RFP_1
from ..layers.stem.SharLayer import SharLayer
import math
class MyHead(nn.Module):
    def __init__(
        self,
        num_classes=1,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False
    ):
        """
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels=in_channels
        self.strides=strides
        """
        the stems to change the channels
        """
        self.stems=nn.ModuleList()
        """
        对reg进行预测
        reg说呢工程调整数据
        iou用于判断是否可应作为目标
        """
        self.reg_convs=nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.iou_preds = nn.ModuleList()
        """
        用于判断是否有目标
        """
        self.cls_convs=nn.ModuleList()
        self.cls_preds=nn.ModuleList()
        """
        mask
        """
        self.mas_convs=nn.ModuleList()
        self.mas_preds=nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv
        shifts_list=[[3,5,7,9],[2,4,6,8],[1,2,3,4]]
        for i in range(len(self.in_channels)):
            self.stems.append(
                nn.Sequential(*[
                    SharLayer(
                        in_channels=int(in_channels[i] * width),
                        out_channels=int(256 * width),
                        expansion=0.5,
                        act=act)
                ])
            )

            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.mas_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.iou_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.mas_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.mas_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.iou_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def forward(self,xin):
        outputs=[]
        for k,(stride,x) in enumerate(zip(self.strides,xin)):
            nx=self.stems[k](x)
            cls_x=nx
            reg_x=nx
            mas_x=nx
            cls_feat=self.cls_convs[k](cls_x)
            reg_feat=self.reg_convs[k](reg_x)
            mas_feat=self.mas_convs[k](mas_x)
            cls_pred=self.cls_preds[k](cls_feat)
            reg_pred=self.reg_preds[k](reg_feat)
            iou_pred=self.iou_preds[k](reg_feat)
            mas_pred=self.mas_preds[k](mas_feat)
            output=torch.concat([reg_pred,mas_pred,iou_pred,cls_pred],dim=1)
            batch_size,num_o,_,_=output.shape
            output=output.permute(0,2,3,1).view(batch_size,-1,num_o).contiguous()
            outputs.append(output)
        return outputs
