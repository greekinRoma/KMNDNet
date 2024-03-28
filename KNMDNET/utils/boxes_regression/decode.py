import math
import torch
class Box2BoxTransform(object):
    def __init__(self):
        super().__init__()
    def get_deltas(self,
                   src_boxes:torch.Tensor,
                   target_boxes:torch.Tensor):
        """

        Args:
            src_boxes:
            target_boxes:

        Returns:

        """
        assert isinstance(src_boxes,torch.Tensor),type(src_boxes)
        assert  isinstance(target_boxes,torch.Tensor),type(target_boxes)
        dx=(target_boxes[...,0]-src_boxes[...,0])/src_boxes[...,2]
        dy=(target_boxes[...,1]-src_boxes[...,1])/src_boxes[...,3]
        dw=torch.log(target_boxes[...,2]/src_boxes[...,2])
        dh=torch.log(target_boxes[...,3]/src_boxes[...,3])
        deltas=torch.stack([dx,dy,dw,dh],dim=-1)
        return deltas