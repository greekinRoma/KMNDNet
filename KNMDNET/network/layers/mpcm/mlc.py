import torch
from torch import nn
class MLC(nn.Module):
    def __init__(self,shifts=[2,4]):
        super().__init__()
        self.shifts=shifts
    def circ_shift(self,cen, shift):
        #对特征图进行平移，为了方便计算不同位置的对比度
        _, _, hei, wid = cen.shape
        ######## B1 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B1_NW = cen[:, :, shift:, shift:]          # B1_NW is cen's SE
        B1_NE = cen[:, :, shift:, :shift]      # B1_NE is cen's SW
        B1_SW = cen[:, :, :shift, shift:]      # B1_SW is cen's NE
        B1_SE = cen[:, :, :shift, :shift]          # B1_SE is cen's NW
        B1_N = torch.concat([B1_NW, B1_NE], dim=3)
        B1_S = torch.concat([B1_SW, B1_SE], dim=3)
        B1 = torch.concat([B1_N, B1_S], dim=2)
        ######## B2 #########
        # old: A  =>  new: B
        #      B  =>       A
        B2_N = cen[:, :, shift:, :]          # B2_N is cen's S
        B2_S = cen[:, :, :shift, :]      # B2_S is cen's N
        B2 = torch.concat([B2_N, B2_S], dim=2)
        ######## B3 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B3_NW = cen[:, :, shift:, wid-shift:]          # B3_NW is cen's SE
        B3_NE = cen[:, :, shift:, :wid-shift]      # B3_NE is cen's SW
        B3_SW = cen[:, :, :shift, wid-shift:]      # B3_SW is cen's NE
        B3_SE = cen[:, :, :shift, :wid-shift]          # B1_SE is cen's NW
        B3_N = torch.concat([B3_NW, B3_NE], dim=3)
        B3_S = torch.concat([B3_SW, B3_SE], dim=3)
        B3 = torch.concat([B3_N, B3_S], dim=2)
        ######## B4 #########
        # old: AB  =>  new: BA
        B4_W = cen[:, :, :, wid-shift:]          # B2_W is cen's E
        B4_E = cen[:, :, :, :wid-shift]          # B2_E is cen's S
        B4 = torch.concat([B4_W, B4_E], dim=3)
        ######## B5 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B5_NW = cen[:, :, hei-shift:, wid-shift:]          # B5_NW is cen's SE
        B5_NE = cen[:, :, hei-shift:, :wid-shift]      # B5_NE is cen's SW
        B5_SW = cen[:, :, :hei-shift, wid-shift:]      # B5_SW is cen's NE
        B5_SE = cen[:, :, :hei-shift, :wid-shift]          # B5_SE is cen's NW
        B5_N = torch.concat([B5_NW, B5_NE], dim=3)
        B5_S = torch.concat([B5_SW, B5_SE], dim=3)
        B5 = torch.concat([B5_N, B5_S], dim=2)
        ######## B6 #########
        # old: A  =>  new: B
        #      B  =>       A
        B6_N = cen[:, :, hei-shift:, :]          # B6_N is cen's S
        B6_S = cen[:, :, :hei-shift, :]      # B6_S is cen's N
        B6 = torch.concat([B6_N, B6_S], dim=2)
        ######## B7 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B7_NW = cen[:, :, hei-shift:, shift:]          # B7_NW is cen's SE
        B7_NE = cen[:, :, hei-shift:, :shift]      # B7_NE is cen's SW
        B7_SW = cen[:, :, :hei-shift, shift:]      # B7_SW is cen's NE
        B7_SE = cen[:, :, :hei-shift, :shift]          # B7_SE is cen's NW
        B7_N = torch.concat([B7_NW, B7_NE], dim=3)
        B7_S = torch.concat([B7_SW, B7_SE], dim=3)
        B7 = torch.concat([B7_N, B7_S], dim=2)
        ######## B8 #########
        # old: AB  =>  new: BA
        B8_W = cen[:, :, :, shift:]          # B8_W is cen's E
        B8_E = cen[:, :, :, :shift]          # B8_E is cen's S
        B8 = torch.concat([B8_W, B8_E], dim=3)
        return B1, B2, B3, B4, B5, B6, B7, B8
        #计算得到图像转化的特征图，这是为了方便计算对比度
        #计算不同区域对比度的最小值
        #这样可以减小图像特例的干扰
        #也就是我们计算中有用的值
    def cal_pcm(self,cen, shift):
        B1, B2, B3, B4, B5, B6, B7, B8 = self.circ_shift(cen, shift=shift)
        s1 = (B1 - cen) * (B5 - cen)
        s2 = (B2 - cen) * (B6 - cen)
        s3 = (B3 - cen) * (B7 - cen)
        s4 = (B4 - cen) * (B8 - cen)
        #将不同的特征层进行计算
        #计算图像区块的变化
        c12 = torch.minimum(s1, s2)
        c123 = torch.minimum(c12, s3)
        c1234 = torch.minimum(c123, s4)
        return c1234
    def forward(self,input):
        tmps=[]
        for shift in self.shifts:
            tmps.append(self.cal_pcm(cen=input,shift=shift))
        tmps=torch.stack(tmps,-1)
        out,_=torch.max(tmps,-1)
        return out