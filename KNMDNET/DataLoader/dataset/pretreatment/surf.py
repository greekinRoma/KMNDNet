import cv2
import numpy as np
import itertools
def surf(src):
    kernel=1
    # src=cv2.blur(src,(kernel,kernel))
    # if len(src.shape)<3:
    #     src=np.expand_dims(src,-1)
    shift=2
    middle_index=(2*shift+1)*(2*shift+1)//2
    tmps=[]
    for i,j in itertools.product((np.arange(2*shift+1)-shift)*kernel,repeat=2):
        tmp=np.roll(src,i,0)
        tmp=np.roll(tmp,j,1)
        tmps.append(tmp)
    tmps=np.concatenate(tmps,2)
    index=np.argsort(tmps,2)[:,:,middle_index:middle_index+1]/25.*255.
    return index