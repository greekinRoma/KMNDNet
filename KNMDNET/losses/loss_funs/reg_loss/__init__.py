from losses.loss_funs.reg_loss.iou_loss.iouloss import IOUloss
from losses.loss_funs.reg_loss.iou_loss.diouloss import DIOUloss
from losses.loss_funs.reg_loss.iou_loss.giouloss import GIOUloss
from losses.loss_funs.reg_loss.iou_loss.ciouloss import CIOUloss
from .smooth_l1 import SmoothL1Loss
def GetIouLoss(name:str="iou_loss",reduction:str="none"):
    print(name)
    if name not in ["iou_loss","giou_loss","diou_loss","ciou_loss"]: raise r"{} is not in [iou_loss,giou_loss,diou_loss,giou_loss]".format(name)
    if (name=="iou_loss"):
        return IOUloss(reduction)
    elif(name=="giou_loss"):
        return GIOUloss(reduction)
    elif(name=="diou_loss"):
        return DIOUloss(reduction)
    elif (name=="ciou_loss"):
        return CIOUloss(reduction)
    else:
        return IOUloss(reduction)
