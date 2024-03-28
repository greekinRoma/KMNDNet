from torch import nn
import torch
from utils import meshgrid
def init_grids(outputs,strides,dtype):
    grids=[]
    for output,stride in zip(outputs,strides):
        hsize,wsize=output.shape[2:4]
        yv,xv=meshgrid([torch.arange(hsize),torch.arange(wsize)])
        grid=torch.stack((xv,yv),2).view(1,1,hsize,wsize,2).type(dtype)
        grid=grid.view(1,-1,2)
        grids.append(grid)
    return grids