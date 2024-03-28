import cv2
import numpy as np
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
def tophat(src):
    dst=cv2.morphologyEx(src,cv2.MORPH_TOPHAT,kernel)
    dst=np.expand_dims(dst,-1)
    # cv2.imshow("outcome",dst)
    # cv2.waitKey(0)
    return dst