from utils import compute_iou
import numpy as np
import matplotlib.pyplot as plt
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
class VOCEvaluator:
    def __init__(self,ovthreshes):
        self.ovthreshes=ovthreshes
        self.num_thre = len(ovthreshes)
        self.scores=[[] for _ in range(self.num_thre)]
        self.labels=[[] for _ in range(self.num_thre)]
        self.gts=0
    def push(self,boxes,scores,targets):
        print(scores)
        print(boxes)
        iou=compute_iou(targets,boxes)
        max_iou=np.max(iou,1)
        max_jax=np.argmax(iou,1)
        for i,ovthresh in enumerate(self.ovthreshes):
            mask=max_iou>ovthresh
            index=max_jax[mask]
            print(max_iou)
            print(index)
            labels = np.zeros(len(boxes), dtype=np.uint8)
            labels[index]=1.0
            self.labels[i].append(labels)
            self.scores[i].append(scores)
        self.gts+=len(targets)
    def compute_AP(self):
        pres=[]
        recs = []
        for labels,scores in zip(self.labels,self.scores):
            scores=np.concatenate(scores,-1)
            labels=np.concatenate(labels,-1)
            index=np.argsort(-scores)
            tp=labels[index]
            fp=1.0-tp
            fp=np.cumsum(fp)
            tp=np.cumsum(tp)
            pre=tp/np.maximum(tp + fp, np.finfo(np.float64).eps)
            rec=tp/self.gts
            f = open('../../datasets/eval.txt', 'a')
            for p, r in zip(range(len(scores)), fp):
                f.write(f"{p},{r},")
            f.write('\n')
            f.close()
            pres.append(pre)
            recs.append(rec)
        return pres,recs
if __name__=='__main__':
    a=[1,2,3,4,5]
    print(np.argsort(a))