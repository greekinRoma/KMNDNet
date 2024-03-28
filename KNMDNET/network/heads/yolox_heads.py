import torch
import torch.nn as nn
from ..network_blocks import BaseConv, DWConv
from ..layers.Shar_Layer import shar_layer
import math
from setting.read_setting import config as cfg
from ..layers.lmpcm.lca_14 import ExpansionContrastModule
class YOLOHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False
        self.strides=strides
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.mask_preds=nn.ModuleList()
        self.mask_convs=nn.ModuleList()
        self.shar_stems=nn.ModuleList()
        self.lca_layers=nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv
        for i in range(len(in_channels)):
            self.shar_stems.append(
                shar_layer(
                    in_channels=int(in_channels[i]*width),
                    out_channels=int(256*width),
                    expansion=0.5,
                    act=act
                )
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
            self.mask_convs.append(
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


            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(ExpansionContrastModule(in_channels=int(width*256),out=1))
            self.cls_preds.append(ExpansionContrastModule(in_channels=int(width*256),out=1))
            self.mask_preds.append(
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
            try:
                b = conv.bias.view(1, -1)
                b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except:
                conv.initialize_biases(prior_prob)
        for conv in self.obj_preds:
            try:
                b = conv.bias.view(1, -1)
                b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except:
                conv.initialize_biases(prior_prob)
        for conv in self.lca_layers:
            conv.initialize_biases(prior_prob)
    def forward(self,xin):
        outputs=[]
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            nx = self.shar_stems[k](x)
            mas_x = nx
            cls_x = nx
            reg_x = nx
            cls_feat = cls_conv(cls_x)
            reg_feat = reg_conv(reg_x)
            mask_feat = self.mask_convs[k](mas_x)
            mask_output = self.mask_preds[k](mask_feat)
            cls_output = self.cls_preds[k](cls_feat)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            output=torch.cat([reg_output,obj_output,cls_output,mask_output],1)
            batch_size = output.shape[0]
            num_o = output.shape[1]
            outputs.append(output.permute(0,2,3,1).view(batch_size,-1,num_o))
        return outputs