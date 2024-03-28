from .mynet_loss import MyNetLoss
from .yolox_loss import YOLOXLoss
def get_loss_type(mode:str="mynet_loss"):
    mode_names=["mynet","yolox_s"]
    loss_funcs=[MyNetLoss,YOLOXLoss]
    assert mode in mode_names ,r"{} is not in ".format(mode,loss_funcs)
    index=mode_names.index(mode)
    return loss_funcs[index]
