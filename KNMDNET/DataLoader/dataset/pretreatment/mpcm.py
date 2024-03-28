import numpy as np
def mpcm(src):
    def circ_shift(cen, shift):
        #对特征图进行平移，为了方便计算不同位置的对比度
        hei, wid,_ = cen.shape
        ######## B1 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B1_NW = cen[ shift:, shift:,:]          # B1_NW is cen's SE
        B1_NE = cen[ shift:, :shift,:]      # B1_NE is cen's SW
        B1_SW = cen[:shift, shift:,:]      # B1_SW is cen's NE
        B1_SE = cen[:shift, :shift,:]          # B1_SE is cen's NW
        B1_N = np.concatenate([B1_NW, B1_NE], axis=1)
        B1_S = np.concatenate([B1_SW, B1_SE], axis=1)
        B1 = np.concatenate([B1_N, B1_S], axis=0)
        ######## B2 #########
        # old: A  =>  new: B
        #      B  =>       A
        B2_N = cen[shift:, :,:]          # B2_N is cen's S
        B2_S = cen[:shift, :,:]      # B2_S is cen's N
        B2 = np.concatenate([B2_N, B2_S], axis=0)
        ######## B3 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B3_NW = cen[shift:, wid-shift:, :]          # B3_NW is cen's SE
        B3_NE = cen[ shift:, :wid-shift,:]      # B3_NE is cen's SW
        B3_SW = cen[:shift, wid-shift:,:]      # B3_SW is cen's NE
        B3_SE = cen[:shift, :wid-shift,:]          # B1_SE is cen's NW
        B3_N = np.concatenate([B3_NW, B3_NE], axis=1)
        B3_S = np.concatenate([B3_SW, B3_SE], axis=1)
        B3 = np.concatenate([B3_N, B3_S], axis=0)
        ######## B4 #########
        # old: AB  =>  new: BA
        B4_W = cen[ :, wid-shift:,:]          # B2_W is cen's E
        B4_E = cen[:, :wid-shift,:]          # B2_E is cen's S
        B4 = np.concatenate([B4_W, B4_E], axis=1)
        ######## B5 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B5_NW = cen[hei-shift:, wid-shift:,:]          # B5_NW is cen's SE
        B5_NE = cen[hei-shift:, :wid-shift,:]      # B5_NE is cen's SW
        B5_SW = cen[:hei-shift, wid-shift:,:]      # B5_SW is cen's NE
        B5_SE = cen[:hei-shift, :wid-shift,:]          # B5_SE is cen's NW
        B5_N = np.concatenate([B5_NW, B5_NE], axis=1)
        B5_S = np.concatenate([B5_SW, B5_SE], axis=1)
        B5 = np.concatenate([B5_N, B5_S], axis=0)
        ######## B6 #########
        # old: A  =>  new: B
        #      B  =>       A
        B6_N = cen[hei-shift:, :,:]          # B6_N is cen's S
        B6_S = cen[:hei-shift, :,:]      # B6_S is cen's N
        B6 = np.concatenate([B6_N, B6_S], axis=0)
        ######## B7 #########
        # old: AD  =>  new: CB
        #      BC  =>       DA
        B7_NW = cen[hei-shift:, shift:,:]          # B7_NW is cen's SE
        B7_NE = cen[hei-shift:, :shift,:]      # B7_NE is cen's SW
        B7_SW = cen[:hei-shift, shift:,:]      # B7_SW is cen's NE
        B7_SE = cen[:hei-shift, :shift,:]          # B7_SE is cen's NW
        B7_N = np.concatenate([B7_NW, B7_NE], axis=1)
        B7_S = np.concatenate([B7_SW, B7_SE], axis=1)
        B7 = np.concatenate([B7_N, B7_S], axis=0)
        ######## B8 #########
        # old: AB  =>  new: BA
        B8_W = cen[:, shift:,:]          # B8_W is cen's E
        B8_E = cen[:, :shift,:]          # B8_E is cen's S
        B8 = np.concatenate([B8_W, B8_E], axis=1)
        return B1, B2, B3, B4, B5, B6, B7, B8
    def cal_pcm(cen, shift):
        B1, B2, B3, B4, B5, B6, B7, B8 = circ_shift(cen, shift=shift)
        s1 = (B1 - cen) * (B5 - cen)
        s2 = (B2 - cen) * (B6 - cen)
        s3 = (B3 - cen) * (B7 - cen)
        s4 = (B4 - cen) * (B8 - cen)
        #将不同的特征层进行计算
        #计算图像区块的变化
        c12 = np.minimum(s1, s2)
        c123 = np.minimum(c12, s3)
        c1234 = np.minimum(c123, s4)
        return c1234
    tmps=[]
    for shift in [1,3,5,7]:
        tmps.append(cal_pcm(cen=src,shift=shift))
    tmps=np.stack(tmps,-1)
    dst=np.max(tmps,-1)
    dst=dst/np.max(dst)*255.
    return dst