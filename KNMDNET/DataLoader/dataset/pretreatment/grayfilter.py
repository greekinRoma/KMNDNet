import numpy as np
def grayfilter(src):
    src=np.mean(src,-1,keepdims=True)
    return src