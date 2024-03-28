import numpy as np
def compute_iou(box_a,box_b):
    box_a=np.expand_dims(box_a,1)
    box_b=np.expand_dims(box_b,0)
    xymin=np.maximum(box_a[...,:2],box_b[...,:2])
    xymax=np.minimum(box_a[...,2:],box_b[...,2:])
    wh=np.maximum(xymax-xymin+1.,0.0)
    inter_area=wh[...,0]*wh[...,1]
    iou=inter_area/((box_a[...,3]-box_a[...,1]+1.)*(box_a[...,2]-box_a[...,0]+1.)
    +(box_b[...,3]-box_b[...,1]+1.)*(box_b[...,2]-box_b[...,0]+1.)-inter_area)
    return iou
if __name__=='__main__':
    box_a=np.array([[0,0,1,1],[0,0,2,2]])
    targets=np.array([[0,0,1/2,1/2],[0,0,2,2],[1,1,2,2]])
    iou=compute_iou(targets,box_a)
    print(iou)
    print(np.max(iou,0))