import random

import numpy as np


def _mirror(image, boxes,prob):
    hight, width, _ = image.shape
    if random.random() <prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    if random.random()<prob:
        image = image[::-1]
        boxes[:, 1::2] = hight - boxes[:, 3::-2]
    return image, boxes
def _mirror_img(image,prob):
    if random.random() <prob:
        image = image[:, ::-1]
    if random.random()<prob:
        image = image[::-1]
    return image
def _transpose_img(image,w,h,prob):
    if random.random()<prob:
        image=np.transpose(image,(1,0,2))
    return image,w,h