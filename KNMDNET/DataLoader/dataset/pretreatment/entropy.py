import itertools

import numpy as np
import cv2
index=np.eye(8)
def entropy(src):
    kernel=2
    src=cv2.blur(src,(kernel,kernel))
    src=np.expand_dims(src,-1)
    show_src = src
    src=src//32
    shift=3
    tmps=[]
    for i,j in itertools.product((np.arange(2*shift+1)-shift)*kernel,repeat=2):
        tmp=np.roll(src,i,0)
        tmp=np.roll(tmp,j,1)
        tmps.append(tmp)
        # print("{},{}".format(i,j))
        # cv2.imshow("tmp",tmp*32/255)
        # cv2.waitKey(0)
    tmps=np.concatenate(tmps,2)
    tmps_0=np.sum(tmps==0,-1,keepdims=True)
    tmps_1=np.sum(tmps==1,-1,keepdims=True)
    tmps_2=np.sum(tmps==2,-1,keepdims=True)
    tmps_3 = np.sum(tmps == 3, -1, keepdims=True)
    tmps_4 = np.sum(tmps == 4, -1, keepdims=True)
    tmps_5 = np.sum(tmps == 5, -1, keepdims=True)
    tmps_6 = np.sum(tmps == 6, -1, keepdims=True)
    tmps_7 = np.sum(tmps == 7, -1, keepdims=True)
    tmps=np.concatenate([tmps_0,tmps_1,tmps_2,tmps_3,tmps_4,tmps_5,tmps_6,tmps_7],axis=2)/25
    dst=np.sum(-tmps*np.log(tmps+1e-2),axis=2,keepdims=True)
    # cv2.imshow("src", show_src / 255.)
    # cv2.imshow("dst",dst/np.max(dst))
    # cv2.waitKey(0)
    return dst