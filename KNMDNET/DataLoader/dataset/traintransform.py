import numpy as np
from DataLoader.dataset.label_filter.filters import filter_boxes
from utils import xyxy2cxcywh
import cv2
def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

class TrainTransform:
    def __init__(self, max_labels=50, swap=(2,0,1)):
        self.max_labels = max_labels
        self.swap=swap
    def __call__(self, image, targets, input_dim):
        boxes = targets[:, 1:].copy()
        labels = targets[:, 0].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim,self.swap)
            return image, targets
        #image中的target的数量，如果数量过多，就直接输出，节约计算量
        height, width, _ = image.shape
        image_t, _ = preproc(image, input_dim,self.swap)
        boxes=np.clip(boxes,0,input_dim[0])
        boxes = xyxy2cxcywh(boxes)
        boxes_t,labels_t=filter_boxes(boxes,labels)
        #过滤掉过度失真的数据
        if len(boxes_t) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            return image_t, targets
        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[:self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels