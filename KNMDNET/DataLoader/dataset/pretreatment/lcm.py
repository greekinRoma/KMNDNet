import numpy as np
import cv2


def max_filter(img, K_size):
    height, width,_ = img.shape
    pad = K_size // 2
    out_img = img.copy()
    pad_img = np.zeros((height + pad * 2, width + pad * 2,1), dtype=np.uint8)
    pad_img[pad: pad + height, pad: pad + width,:] = img.copy()

    for y in range(height):
        for x in range(width):
            out_img[y, x] = np.max(pad_img[y:y + K_size, x:x + K_size])
    return out_img


def lcm(src):
    kernel_size=5
    background=(cv2.blur(src,(kernel_size*3,kernel_size*3))*9-cv2.blur(src,(kernel_size,kernel_size)))/8
    background=np.expand_dims(background,-1)
    target=max_filter(src,kernel_size)
    return target/(background+1)