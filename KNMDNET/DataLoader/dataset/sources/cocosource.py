#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import copy
import os
from ..pretreatment import transform
import cv2
import numpy as np
from pycocotools.coco import COCO
from DataLoader.wrapper import CacheDataset,cache_read_img
def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


class COCOSource(CacheDataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        mode='test',
        name="train2017",
        img_size=(640, 640),
        cache=False,
        cache_type="ram",
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        assert mode in ['test', 'train', 'trainval', 'val','total'], print(
            r'mode must be test,train,trainval or val,but the input of the mode is {}'.format(mode))
        self.mode=mode
        self.data_dir = os.path.join(data_dir,"COCO")
        self.json_file = r'{}.json'.format(mode)
        self.coco = COCO(os.path.join(self.data_dir, "Annotations", self.json_file))
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.index_list=np.arange(len(self.ids))
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.name = name
        self.img_size = img_size
        self.annotations = self._load_coco_annotations()
        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        self.file_names=[anno[3] for anno in self.annotations]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )
    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.index_list]
    def send_ids(self):
        return self.file_names
    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs([self.ids[id_]])[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[self.ids[id_]], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                                 interpolation=cv2.INTER_LINEAR, ).astype(np.float32)
        resized_img = transform(types=['lmpcm'], src=resized_img)
        return resized_img
    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir,'JPEGImages',file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)
    def __len__(self):
        return len(self.ids)
    def _input_dim(self):
        return self.img_size
    def pull_item(self, index):
        id_ = self.file_names[index]
        label, origin_image_size, _, _ = self.annotations[index]
        label=np.concatenate([np.ones([len(label),1]),label[...,:4]],-1)
        img = self.read_img(index)
        name=id_
        return img, copy.deepcopy(label), name,origin_image_size, np.array([index])
    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, name, img_info, img_id = self.pull_item(index)
        return img, target, name, img_info, img_id
if __name__=='__main__':
    source=COCOSource(data_dir=r'C:\Users\27227\Desktop\datasets\ISDD\data')
    for i in range(len(source)):
        img, target, name, img_info, img_id=source[i]
        print(target)
        for t in target:
            cv2.rectangle(img,(int(t[1]),int(t[2])),(int(t[3]),int(t[4])),color=(0,255,0))
        cv2.imshow('outcome', img)
        cv2.waitKey(0)