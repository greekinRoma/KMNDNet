import numpy as np
import cv2
from utils import *
def vision(name,imgs):
    for batch_img in imgs:
        for i,img in enumerate(batch_img):
            img=np.ascontiguousarray(img.cpu())
            cv2.imshow(name,img/255.)
            cv2.waitKey(0)
def vision_outcome(name,labels,imgs):
    for img,label in zip(imgs,labels):
        label=label[label[...,0]>0]
        cimg=img.cpu().numpy().copy()
        cimg=np.transpose(cimg,[1,2,0])
        cimg=np.ascontiguousarray(cimg)
        label[...,1:]=cxcywh2xyxy(label[...,1:])
        for l in label:
            cv2.rectangle(cimg,[int(l[1]),int(l[2])],[int(l[3]),int(l[4])],color=(0,255,255))
        cv2.imshow(name,cimg/255.)
        cv2.waitKey(0)