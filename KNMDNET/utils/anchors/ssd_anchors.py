import torch
from torch import nn
class SSDAnchors(object):
    def __init__(self):
        super(SSDAnchors,self).__init__()
    def forward(self,inp):
        mean=[]