from utils.iou.iou import IOU
class IOUloss(IOU):
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction
    def forward(self, pred, target):
        assert pred.size()==target.size() and target.numel()>0
        pred = pred.view(-1, 4)#将输出拉直
        target = target.view(-1, 4)#将先验拉直
        iou = super().get_iou(pred, target)
        loss = 1 - iou**2
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
