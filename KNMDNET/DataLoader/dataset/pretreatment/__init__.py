import numpy
import numpy as np
from .grayfilter import *
from .mpcm import *
from .lbp import *
from .canny import *
from .sobel import *
from .entropy import *
from .lmpcm import *
from .surf import *
from .lmp import *
from .lcm import *
import numpy as np
def transform(types:list,src:numpy.array):
    gray_img=grayfilter(src)
    names=['mpcm','lbp','canny','sobel','entropy','lmpcm','surf','lmp','lcm']
    filters=[mpcm,lbp,canny,sobel,entropy,lmpcm,surf,lmp,lcm]
    tmps=[]
    for type in types:
        assert type in names, "{} is not in names".format(type)
        index=names.index(type)
        tmps.append(filters[index](gray_img))
    rest_num=2-len(tmps)
    tmps.append(np.repeat(gray_img,rest_num,-1))
    assert rest_num>=0, "the length of types is 3"
    return np.concatenate(tmps,-1)