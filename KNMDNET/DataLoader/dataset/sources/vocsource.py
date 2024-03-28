import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from DataLoader.wrapper import CacheDataset,cache_read_img
from ..pretreatment import transform
class VocSource(CacheDataset):
    def __init__(self,data_dir,cache_type="ram",cache=True,preproc=None,mode='test',img_size=(640,640)):
        assert mode in ['test','train','trainval','val'],print(r'mode must be test,train,trainval or val,but the input of the mode is {}'.format(mode))
        self.data_dir=data_dir
        self.name=mode
        self.cache=cache
        self.anno_dir=os.path.join(data_dir,'Annotations')
        self.img_dir=os.path.join(data_dir,'JPEGImages')
        self.ids_file=os.path.join(data_dir,r'ImageSets/Main/{}.txt'.format(mode))
        self.ids=self.get_labels(self.ids_file)
        self.num_imgs=len(self.ids)
        if img_size is None:
            name=self.ids[0]
            img_path = os.path.join(self.img_dir, name + '.png')
            img = cv2.imread(img_path)
            img_size=img.shape[:2]
        self.img_size=img_size
        self.keep_difficult=False
        self.preproc=preproc
        path_filename = [os.path.join(self.img_dir,ids) for ids in self.ids]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=self.data_dir,
            cache_dir_name=f"cache_{self.name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type)
    def get_labels(self,label_file):
        f=open(label_file,'r')
        ids=f.read().strip().split('\n')
        return ids
    def send_ids(self):
        return self.ids
    def __len__(self):
        return len(self.ids)
    def _input_dim(self):
        return self.img_size
    @cache_read_img(use_cache=True)
    def read_img(self,idx):
        name=self.ids[idx]
        img_path = os.path.join(self.img_dir, name + '.png')
        img = cv2.imread(img_path)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(img,(int(img.shape[1] * r), int(img.shape[0] * r)),interpolation=cv2.INTER_LINEAR,).astype(np.float32)
        resized_img=transform(types=['mpcm','lbp'],src=resized_img)
        # print(resized_img.shape)
        # cv2.imshow("outcome",resized_img/255.)
        # cv2.waitKey(0)
        return resized_img
    def target_transform(self,target):
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None:
                difficult = int(difficult.text) == 1
            else:
                difficult = False
            if not self.keep_difficult and difficult:
                continue
            bbox = obj.find("bndbox")
            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                bndbox.append(cur_pt)
            label_idx = 1
            bndbox.insert(0,label_idx)
            res = np.vstack((res, bndbox))
        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_info = (height, width)
        return res, img_info
    def load_anno(self,idx):
        name = self.ids[idx]
        anno_path = os.path.join(self.anno_dir, name + '.xml')
        target = ET.parse(anno_path).getroot()
        res, img_info = self.target_transform(target)
        height, width = img_info
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, 1:] *= r
        resized_info = (int(height * r), int(width * r))
        return (res, img_info, resized_info)
    def pull_item(self, index):
        img=self.read_img(index)
        target,img_info,_=self.load_anno(index)
        name=self.ids[index]
        return img,target,name,img_info,index
    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, name, img_info, img_id = self.pull_item(index)
        return img, target, name, img_info, img_id
if __name__=='__main__':
    data_dir=r'../../../../datasets/ISDD/VOC2007'
    source=VocSource(data_dir=r'C:\Users\27227\Desktop\datasets\ISDD\data\VOCdevkit2007\VOC2007')
    for i in range(len(source)):
        img, target, name, img_info, img_id=source[i]
        print(target)
        cv2.imshow('outcome',img)
        cv2.waitKey(0)