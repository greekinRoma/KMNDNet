import cv2
import numpy as np
import torch
def show_ota(img,output,target,strides,grids,fg_mask,matched_gt_inds):
    img=torch.permute(img,(1,2,0))
    img=np.ascontiguousarray(img.cpu()).astype(np.uint8)
    for p, g, coor in zip(output[fg_mask, :4], target[matched_gt_inds, 1:],
                          ((grids + 0.5) * strides.unsqueeze(-1))[fg_mask.unsqueeze(0)]):
        cv2.rectangle(img, (int(p[0] - p[2] / 2), int(p[1] - p[3] / 2)), (int(p[0] + p[2] / 2), int(p[1] + p[3] / 2)),
                      (255, 0, 0))
        cv2.rectangle(img, (int(g[0] - g[2] / 2), int(g[1] - g[3] / 2)), (int(g[0] + g[2] / 2), int(g[1] + g[3] / 2)),
                      (0, 0, 255))
        cv2.circle(img, (int(coor[0]), int(coor[1])), 2, (0, 255, 0), 2)
    cv2.imshow("outcome", img)
    cv2.waitKey(0)