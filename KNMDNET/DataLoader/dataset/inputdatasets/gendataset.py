import random
import numpy as np
from DataLoader.dataset.sources.vocsource import VocSource
from DataLoader.dataset.sources.targetsource import TargetSource
from utils.mirror import _mirror_img
from utils.iou import compute_iou


class GenDataset():
    def __init__(self,
                 prob:float,
                 maxtarget:int,
                 vocsource:VocSource,
                 targetsource:TargetSource):
        self.maxtarget=maxtarget
        self.vocsource=vocsource
        self.img_size = self.vocsource.img_size
        self.targetsource=targetsource
        self.num_target=len(targetsource)
        self.prob=prob
        self.w,self.h=self.vocsource.img_size
    def __len__(self):
        return len(self.vocsource)
    def _input_dim(self):
        return self.img_size
    def __getitem__(self, item):
        img, target, name, img_info, img_id=self.vocsource[item]
        h,w,_=img.shape
        num_target=random.randint(1,self.maxtarget)
        ids=[random.randint(0,self.num_target-1) for i in range(num_target)]
        target_imgs=[]
        target_labels=[]
        labels=[]
        for i in ids:
            target_img,w,h=self.targetsource[i]
            target_img=_mirror_img(target_img,prob=self.prob)
            if random.random()<self.prob:
                target_img=np.transpose(target_img,(1,0,2))
                k=h
                h=w
                w=k
            target_imgs.append(target_img)
            cx=random.randint(1,self.w-w-1)
            cy=random.randint(1,self.h-h-1)
            target_labels.append([1,cx+w/4-1,cy+h/4-1,cx+w*3/4+1,cy+h*3/4+1])
            labels.append([1,int(cx),int(cy),int(cx+w),int(cy+h)])
        labels=np.array(labels)
        target_labels=np.array(target_labels)
        origin_boxes=np.concatenate([labels[...,1:],target[...,1:]],0)
        iou=compute_iou(labels[...,1:],origin_boxes)
        index=np.arange(len(labels))
        iou[index,index]=0
        mask=np.max(iou,-1)<=0.
        if len(labels)==0:
            return img,target,name,img_info,img_id
        for l,target_img,m in zip(labels,target_imgs,mask):
            if m==False:
                continue
            img[l[2]:l[4],l[1]:l[3]]=target_img
        target=np.concatenate([target,target_labels[mask]],0)
        return img,target,name,img_info,img_id
if __name__=="__main__":
    from DataLoader.dataset.traintransform import TrainTransform
    from DataLoader.dataset.sources.vocsource import VocSource
    from VisionTools.show_img import vision_outcome
    from DataLoader.dataset.testdataset import TestDataset
    from DataLoader.dataloader import DataLoader
    voc_source = VocSource(data_dir=r'../../../../datasets/ISDD/VOC2007')
    target_source = TargetSource(data_dir=r'../../../../datasets')
    max_target=10
    gendataset=GenDataset(
        prob=0.5,
        maxtarget=max_target,
        vocsource=voc_source,
        targetsource=target_source
    )
    testdataset=TestDataset(base_dataset=gendataset, preproc=TrainTransform())
    dataloader = DataLoader(dataset=testdataset, batch_size=2, use_cuda=False)
    for imgs, targets, names in dataloader:
        vision_outcome(name='outcome', labels=targets, imgs=imgs)