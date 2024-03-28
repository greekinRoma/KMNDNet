import cv2
import numpy as np
from utils.get_mosaic_coordinate import get_mosaic_coordinate
import random
from DataLoader.dataset.data_augment import random_affine
from DataLoader.dataset.label_filter.filters import filter_xyxy_boxes
class mosaic():
    def __init__(self,
                 degrees:float=10.0,
                 translate:float=0.1,
                 mosaic_scale=(0.5,1.5),
                 shear:float=2.0,
                 input_w:int=640,
                 input_h:int=640,
                 ):
        self.degrees=degrees
        self.translate=translate
        self.scale=mosaic_scale
        self.shear=shear
        self.input_w=input_w
        self.input_h=input_h
    def __call__(self,dataset,mask_dataset,mask,inp,label):
        input_w=self.input_w
        input_h=self.input_h
        num_img=len(dataset)
        indices = [random.randint(0,num_img-1) for _ in range(3)]
        imgs=[inp]
        masks=[mask]
        labels=[label]
        for idx in indices:
            img, label, temp_name, _, img_id =dataset[idx]
            mask,_=mask_dataset[idx]
            imgs.append(img)
            masks.append(mask)
            labels.append(label)
        yc = int(random.uniform(0.5 * self.input_h, 1.5 * self.input_h))
        xc = int(random.uniform(0.5 * self.input_w, 1.5 * self.input_w))
        mosaic_labels=[]
        for i_mosaic,(img,mask,label) in enumerate(zip(imgs,masks,labels)):
            h0, w0 = img.shape[:2]  # orig hw
            scale = min(1. * self.input_h / h0, 1. * self.input_w / w0)
            img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
            (h, w, c) = img.shape[:3]
            if i_mosaic == 0:
                mosaic_img = np.full((self.input_h * 2, self.input_w * 2, c), 114, dtype=np.uint8)
                mosaic_mask = np.full((self.input_h * 2, self.input_w * 2, c), 0, dtype=np.uint8)
            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                mosaic_img, i_mosaic, xc, yc, w, h, self.input_h, self.input_w
            )
            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            mosaic_mask[l_y1:l_y2, l_x1:l_x2] = mask[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1
            labels = label.copy()
            if label.size > 0:
                labels[:, 1] = scale * label[:, 1] + padw
                labels[:, 2] = scale * label[:, 2] + padh
                labels[:, 3] = scale * label[:, 3] + padw
                labels[:, 4] = scale * label[:, 4] + padh
            mosaic_labels.append(labels)
        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            np.clip(mosaic_labels[:, 1], 0, 2 * input_w, out=mosaic_labels[:, 1])
            np.clip(mosaic_labels[:, 2], 0, 2 * input_h, out=mosaic_labels[:, 2])
            np.clip(mosaic_labels[:, 3], 0, 2 * input_w, out=mosaic_labels[:, 3])
            np.clip(mosaic_labels[:, 4], 0, 2 * input_h, out=mosaic_labels[:, 4])
        '''
        mosaic_img, mosaic_labels[..., 1:] = random_affine(
            mosaic_img,
            mosaic_labels[..., 1:],
            target_size=(input_w, input_h),
            degrees=self.degrees,
            translate=self.translate,
            scales=self.scale,
            shear=self.shear,
        )
        '''
        index_mask = filter_xyxy_boxes(mosaic_labels[...,1:])
        mosaic_labels=mosaic_labels[index_mask]
        mosaic_img,mosaic_mask,mosaic_labels[..., 1:] = random_affine(
            img=mosaic_img,
            mask=mosaic_mask,
            targets=mosaic_labels[..., 1:],
            target_size=(input_w, input_h),
            degrees=self.degrees,
            translate=self.translate,
            scales=self.scale,
            shear=self.shear,
        )
        return mosaic_img,mosaic_mask,mosaic_labels
