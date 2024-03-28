import numpy as np
import cv2
from utils import *
import os
def save_outcome(names,save_dir,labels,imgs,need_change):
    for img,label,name in zip(imgs,labels,names):
        label=label[label[...,0]>0]
        cimg=img.cpu().numpy().copy()
        cimg=np.transpose(cimg,[1,2,0])
        cimg=np.ascontiguousarray(cimg)
        if need_change:
            label[...,1:]=cxcywh2xyxy(label[...,1:])
        for l in label:
            cv2.rectangle(cimg,[int(l[1]),int(l[2])],[int(l[3]),int(l[4])],color=(0,255,255))
        cv2.imwrite(os.path.join(save_dir,f'{name}.png'),cimg)