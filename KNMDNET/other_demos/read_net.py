import torch
from torch import nn
module=torch.load("../Evaluator/best_ckpt.pth")['model']
def normize_weight(cen):
        cen = cen ** 2 + 1e-10
        cen = cen / torch.sum(cen, dim=1, keepdim=True)
        return cen
for k,v in module.items():
    if 'scale' in k:
        print(v.squeeze())