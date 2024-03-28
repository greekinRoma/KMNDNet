import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from DataLoader.wrapper import CacheDataset,cache_read_img
class MaskSource(CacheDataset):
    def __init__(self,ids,data_dir,cache_type="ram",cache=True,preproc=None,img_size=(640,640)):
        self.data_dir=data_dir
        self.cache=cache
        self.mask_dir=os.path.join(data_dir,'COCO/Masks')
        self.ids=ids
        self.num_imgs=len(self.ids)
        if img_size is None:
            name=self.ids[0]
            img_path = os.path.join(self.img_dir, name + '.png')
            img = cv2.imread(img_path)
            img_size=img.shape[:2]
        self.img_size=img_size
        self.keep_difficult=False
        self.preproc=preproc
        path_filename = [os.path.join(self.mask_dir,ids) for ids in self.ids]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=self.data_dir,
            cache_dir_name=f"cache_Mask",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type)
    def get_labels(self,label_file):
        f=open(label_file,'r')
        ids=f.read().strip().split('\n')
        return ids
    def __len__(self):
        return len(self.ids)
    def _input_dim(self):
        return self.img_size
    @cache_read_img(use_cache=True)
    def read_img(self,idx):
        name=self.ids[idx]
        img_path = os.path.join(self.mask_dir, name)
        img = cv2.imread(img_path)
        img[img>0]=1.
        img[...,1]=np.minimum(img[...,0]+img[...,1],1)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(img,(int(img.shape[1] * r), int(img.shape[0] * r))).astype(np.uint8)
        return resized_img
    def pull_item(self, index):
        img=self.read_img(index)
        name=self.ids[index]
        return img,name
    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        img, name = self.pull_item(index)
        img=img.astype(float)
        return img, name
if __name__=='__main__':
    data_dir=r'../../../../datasets/ISDD/VOC2007'
    source=MaskSource(data_dir=r'C:\Users\27227\Desktop\datasets\ISDD\data\VOCdevkit2007\VOC2007',cache=True)
    for i in range(len(source)):
        img, name=source[i]
        cv2.imshow('outcome',img*255)
        cv2.waitKey(0)