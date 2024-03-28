import numpy as np
import cv2
def canny(src):
    src = cv2.GaussianBlur(src, (5, 5), 1)
    src=src.astype(np.uint8)
    dst=cv2.Canny(src,50,150)
    dst=dst.astype(np.float32)
    dst=np.expand_dims(dst,-1)
    return dst