import numpy as np
from utils import xyxy2cxcywh
def filter_targets(labels):
    core_labels=labels[:,3:]
    mask=np.min(core_labels,-1)>2
    return labels[mask]
def filter_boxes(boxes,labels):
    tmp_boxes=boxes[:,2:]
    mask = np.min(tmp_boxes, -1) > 4
    return boxes[mask],labels[mask]
def filter_xyxy_boxes(boxes):
    mask=np.min(boxes[...,2:]-boxes[...,:2],-1) > 2
    return mask