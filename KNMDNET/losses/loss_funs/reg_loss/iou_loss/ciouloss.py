from utils.iou.ciou import CIOU
class CIOUloss(CIOU):
    def __init__(self, reduction="none",eps=1e-14):
        super(CIOUloss, self).__init__(eps=eps)
        self.reduction = reduction
    def forward(self, pred, target):
        assert pred.size()==target.size() and target.numel()>0
        pred = pred.view(-1, 4)#将输出拉直
        target = target.view(-1, 4)#将先验拉直
        iou=super().get_iou(pred, target)
        loss=1-iou
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss