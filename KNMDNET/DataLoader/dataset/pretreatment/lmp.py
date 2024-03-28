import numpy as np
def lmp(src):
    def circ_shift(cen, shift):
        hei, wid,_ = cen.shape
        B1_NW = cen[ shift:, shift:,:]
        B1_NE = cen[ shift:, :shift,:]
        B1_SW = cen[:shift, shift:,:]
        B1_SE = cen[:shift, :shift,:]
        B1_N = np.concatenate([B1_NW, B1_NE], axis=1)
        B1_S = np.concatenate([B1_SW, B1_SE], axis=1)
        B1 = np.concatenate([B1_N, B1_S], axis=0)
        B2_N = cen[shift:, :,:]
        B2_S = cen[:shift, :,:]
        B2 = np.concatenate([B2_N, B2_S], axis=0)
        B3_NW = cen[shift:, wid-shift:, :]
        B3_NE = cen[ shift:, :wid-shift,:]
        B3_SW = cen[:shift, wid-shift:,:]
        B3_SE = cen[:shift, :wid-shift,:]
        B3_N = np.concatenate([B3_NW, B3_NE], axis=1)
        B3_S = np.concatenate([B3_SW, B3_SE], axis=1)
        B3 = np.concatenate([B3_N, B3_S], axis=0)
        B4_W = cen[ :, wid-shift:,:]
        B4_E = cen[:, :wid-shift,:]
        B4 = np.concatenate([B4_W, B4_E], axis=1)
        B5_NW = cen[hei-shift:, wid-shift:,:]
        B5_NE = cen[hei-shift:, :wid-shift,:]
        B5_SW = cen[:hei-shift, wid-shift:,:]
        B5_SE = cen[:hei-shift, :wid-shift,:]
        B5_N = np.concatenate([B5_NW, B5_NE], axis=1)
        B5_S = np.concatenate([B5_SW, B5_SE], axis=1)
        B5 = np.concatenate([B5_N, B5_S], axis=0)
        B6_N = cen[hei-shift:, :,:]
        B6_S = cen[:hei-shift, :,:]
        B6 = np.concatenate([B6_N, B6_S], axis=0)
        B7_NW = cen[hei-shift:, shift:,:]
        B7_NE = cen[hei-shift:, :shift,:]
        B7_SW = cen[:hei-shift, shift:,:]
        B7_SE = cen[:hei-shift, :shift,:]
        B7_N = np.concatenate([B7_NW, B7_NE], axis=1)
        B7_S = np.concatenate([B7_SW, B7_SE], axis=1)
        B7 = np.concatenate([B7_N, B7_S], axis=0)
        B8_W = cen[:, shift:,:]
        B8_E = cen[:, :shift,:]
        B8 = np.concatenate([B8_W, B8_E], axis=1)
        return B1, B2, B3, B4, B5, B6, B7, B8
    B1, B2, B3, B4, B5, B6, B7, B8=circ_shift(src,1)
    B1=(B1-src+255)//160
    B2=(B2-src+255)//160
    B3=(B3-src+255)//160
    B4=(B4-src+255)//160
    B5=(B5-src+255)//160
    B6=(B6-src+255)//160
    B7=(B7-src+255)//160
    B8=(B8-src+255)//160
    outs=B1+B2*3+B3*9+B4*27+B5*81+B6*243+B7*729+B8*2187
    return outs