import torch

from DataLoader.dataset.sources.vocsource import VocSource
from DataLoader.dataloader import DataLoader
import cv2
from DataLoader.dataset.traintransform import TrainTransform
from DataLoader.dataset.inputdatasets.basicdatasets import BasicDataset
from DataLoader.dataset.testdataset import TestDataset
import numpy as np
from utils import *
def crop(img,xmin,ymin,xmax,ymax):
    return img[int(ymin):int(ymax),int(xmin):int(xmax)]
def crop_images(img,boxes,itr):
    assert len(img.shape)==3 and len(boxes.shape)==2
    outcomes=[]
    img = np.transpose(img, (1, 2, 0))
    for box in boxes:
        itr=itr+1
        tmp_img=crop(img,box[0],box[1],box[2],box[3])
#        tmp_img=((tmp_img-torch.min(tmp_img))/(torch.max(tmp_img)-torch.min(tmp_img))*255).to(torch.uint8)
        tmp_img = np.ascontiguousarray(tmp_img)
        cv2.imwrite(f'./targets/{itr}.png',tmp_img)
        outcomes.append(tmp_img)
    return outcomes,itr
if __name__=='__main__':
    from setting.read_setting import config as cfg
    from utils import bboxes_iou
    source = VocSource(data_dir=cfg.voc_data_dir,mode='trainval',img_size=None)
    datasets = BasicDataset(source=source, flip_prob=0.0)
    testdataset=TestDataset(base_dataset=datasets, preproc=TrainTransform())
    dataloader=DataLoader(dataset=testdataset,batch_size=1,use_cuda=False)
    size=testdataset.img_size
    itr=0
    for imgs,_,_,targets,names in dataloader:
        target=targets[0]
        #vision_outcome(name='outcome',labels=targets,imgs=imgs)
        mask=target[...,0]>0
        target[..., 3:] = target[..., 3:] * 2
        boxes=cxcywh2xyxy(target[...,1:])
        boxes=boxes[mask]
        iou_mask = bboxes_iou(boxes, boxes, True)
        iou_mask=torch.sum(iou_mask,-1)-1<=0
        boxes=boxes[iou_mask]
        boxes=np.array(boxes)
        mask=np.max(boxes,-1)<=size[0]
        boxes=boxes[mask]
        mask = np.min(boxes, -1) >= 0
        boxes=boxes[mask]
        img=imgs[0]
        outcomes,itr=crop_images(img,boxes,itr=itr)