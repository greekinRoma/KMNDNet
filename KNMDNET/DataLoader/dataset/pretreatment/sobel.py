import numpy as np
import cv2
def sobel(src):
    x_kernel=np.array([[1,2,1],
                       [0,0,0],
                       [-1,-2,-1]])
    y_kernel=np.array([[1,0,-1],
                       [2,0,-2],
                       [1,0,-1]])
    x_dst=cv2.filter2D(src,-1,x_kernel)
    y_dst=cv2.filter2D(src,-1,y_kernel)
    dst=np.sqrt(x_dst**2+y_dst**2)
    dst=np.expand_dims(dst,-1)
    # cv2.imshow("input",src/255.)
    # cv2.imshow("out",dst/255.)
    # cv2.waitKey(0)
    return dst