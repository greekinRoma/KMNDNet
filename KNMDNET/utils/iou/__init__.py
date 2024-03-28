from .ciou import CIOU
from .iou import IOU
from .giou import GIOU
from .diou import DIOU
def get_iou(ioutype:str="None",eps=1e-14):
    if ioutype not in ["ciou","iou","giou","diou"]: raise "{} are not in ciou iou giou and diou".format(ioutype)
    if ioutype=="ciou":
        return CIOU(eps)
    elif ioutype=="iou":
        return IOU(eps)
    elif ioutype=="giou":
        return GIOU(eps)
    elif ioutype=="diou":
        return DIOU(eps)
    else:
        return IOU(eps)