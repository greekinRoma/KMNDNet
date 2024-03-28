import numpy as np
import random
import cv2
import torch

from utils import adjust_box_anns
class mixup():
    def __init__(self,
                 input_w:int,
                 input_h:int,
                 mixup_scale=(0.5,1.5)
                 ):
        self.input_w=input_w
        self.input_h=input_h
        self.mixup_scale=mixup_scale
    def __call__(self,dataset,mask_dataset,origin_img,origin_mask, origin_labels):
        num_img=len(dataset)
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, num_img - 1)
            img, cp_labels, _, _, _ = dataset[cp_index]
        mask,_=mask_dataset[cp_index]
        if len(img.shape) == 3:
            cp_img = np.ones((self.input_h, self.input_w, 3), dtype=np.uint8)*114
            cp_mask = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        else:
            cp_img = np.ones((self.input_h,self.input_w), dtype=np.uint8)*114
            cp_mask = np.zeros((self.input_h, self.input_w), dtype=np.uint8)

        img_cp_scale_ratio = min(self.input_h / img.shape[0], self.input_w / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * img_cp_scale_ratio), int(img.shape[0] * img_cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        mask_cp_scale_ratio = min(self.input_h / mask.shape[0], self.input_w / mask.shape[1])
        resized_mask = cv2.resize(
            mask,
            (int(mask.shape[1] * mask_cp_scale_ratio), int(mask.shape[0] * mask_cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        cp_img[
        : int(img.shape[0] * img_cp_scale_ratio), : int(img.shape[1] * img_cp_scale_ratio)
        ] = resized_img
        cp_mask[
        : int(mask.shape[0] * mask_cp_scale_ratio), : int(mask.shape[1] * mask_cp_scale_ratio)
        ] = resized_mask
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_mask = cv2.resize(
            cp_mask,
            (int(cp_mask.shape[1] * jit_factor), int(cp_mask.shape[0] * jit_factor)),
        )
        img_cp_scale_ratio *= jit_factor
        mask_cp_scale_ratio *= jit_factor
        if FLIP:
            cp_img = cp_img[:, ::-1, :]
            cp_mask = cp_mask[:, ::-1, :]
        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.ones(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )*114
        padded_mask = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img
        padded_mask[:origin_h, :origin_w] = cp_mask
        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
                             y_offset: y_offset + target_h, x_offset: x_offset + target_w
                             ]
        padded_cropped_mask = padded_mask[
                             y_offset: y_offset + target_h, x_offset: x_offset + target_w
                             ]
        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, 1:].copy(), img_cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                    origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        cls_labels = cp_labels[:, 0:1].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((cls_labels, box_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.uint8)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.uint8)
        origin_mask = origin_mask.astype(np.uint8)
        origin_mask = np.maximum(origin_mask,padded_cropped_mask.astype(np.uint8))
        return origin_img.astype(np.uint8), origin_mask,origin_labels