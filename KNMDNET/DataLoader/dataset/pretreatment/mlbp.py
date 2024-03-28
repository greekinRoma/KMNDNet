import numpy as np
def mlbp(src):
    """
    src-(width,height,num_channels)
    dst-(width,height,num_channels)
    """
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
    B1, B2, B3, B4, B5, B6, B7, B8=circ_shift(src,1)
    dst=(B1>src)+2*(B2>src)+4*(B3>src)+8*(B4>src)+16*(B5>src)+32*(B6>src)+64*(B7>src)+128*(B8>src)
    # print(dst.shape)
    return dst