from torch import nn
import torch
def GetAnchors(grid:torch.Tensor,
                stride:torch.Tensor):
    stride=stride.unsqueeze(-1)
    anchors=torch.concatenate([grid[...,:2]*stride,stride,stride],dim=-1)
    return anchors