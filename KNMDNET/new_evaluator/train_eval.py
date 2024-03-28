from new_evaluator.voc_eval import *
import torch
from utils.boxes import postprocess
from tqdm import tqdm
from new_evaluator.coco import coco
class Evaluator():
    def __init__(self,dataloader,need_change):
        super(Evaluator, self).__init__()
        self.dataloader=dataloader
        self.net=None
        self.num_classes=1
        self.test_conf=0.001
        self.nmsthre=0.5
        self.need_change=need_change
        self.coco = coco(save_dir='./', image_set='test', year='2007')
        self.save_file='./content.json'
        self.stats=None
    def model_predict(self,image):
        with torch.no_grad():
            outputs = self.model(image,False)
            outputs[...,4]=outputs[...,4].sigmoid()
            outputs[...,5]=outputs[...,5].sigmoid()
            outputs[...,6]=outputs[...,6].sigmoid()
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.test_conf,
                self.nmsthre,
                class_agnostic=True)
        output=outputs[0]
        if output is None:
            return np.array([]),np.array([])
        return output[..., 0:4].cpu().numpy(),(output[..., 4]*output[..., 5]*output[...,6]).cpu().numpy()
    def push_model(self,model):
        self.model=model
    def eval(self):
        assert self.model is not None,'you should push a model,and the model is None!'
        all_boxes = [[]]
        for i,(imgs,_,_,targets,names) in enumerate(tqdm(self.dataloader)):
            outcome, score = self.model_predict(imgs)
            if len(score)<=0:
                all_boxes[0].append(np.array([[0,0,0,0,0]]))
            else:
                score = np.reshape(score, [-1, 1])
                det = np.concatenate([score, outcome], -1)
                all_boxes[0].append(det)
        self.coco._write_coco_results_file(all_boxes=all_boxes, res_file=self.save_file)
        stats,precision = self.coco._do_detection_eval(res_file=self.save_file,output_dir='./')
        return stats