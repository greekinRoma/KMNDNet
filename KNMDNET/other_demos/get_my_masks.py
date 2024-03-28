import torch
import numpy as np
from setting.read_setting import config as cfg
from DataLoader.dataset.sources.masksource import MaskSource
from DataLoader.dataset.sources.vocsource import VocSource
import cv2
from utils import cxcywh2xyxy,xyxy2cxcywh
from VisionTools.show_img import vision_outcome
data_source = VocSource(data_dir=cfg.voc_data_dir,
                        mode='trainval')
mask_source=MaskSource(ids=data_source.send_ids(),
                        data_dir=r'C:\Users\27227\Desktop\Myexp\other_demos\imgs')
for idx in range(len(data_source)):
    img, target, name, img_info, img_id = data_source[idx]
    mask_img,mask_name=mask_source[idx]
    output_mask=np.zeros([img.shape[0],img.shape[1],3])
    output_mask[...,0]=mask_img[...,0]
    o_target=target.copy()
    target[...,1:]=xyxy2cxcywh(target[...,1:])
    target[...,3:]=target[...,3:]*2
    target[...,1:]=cxcywh2xyxy(target[...,1:])
    #cv2.imshow("outcome",mask_img)
    #cv2.waitKey(0)
    tmp_mask=np.zeros([640,640])
    target[...,1:]=np.clip(target[...,1:],1,639)
    o_target[..., 1:] = np.clip(o_target[..., 1:], 1, 639)
    for t in target:
        tmp_mask[int(t[2])-1:int(t[4])+1,int(t[1])-1:int(t[3])+1]=1
    output_mask[...,1]=tmp_mask
    tmp_mask = np.zeros([640, 640])
    for t in o_target:
        tmp_mask[int(t[2])-1:int(t[4])+1,int(t[1])-1:int(t[3])+1]=1
    output_mask[...,2]=tmp_mask
    cv2.imwrite(r'C:\Users\27227\Desktop\Myexp\other_demos\masks\{}.png'.format(name),output_mask*255)