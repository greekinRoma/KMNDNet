import os.path
import torch
from DataLoader.dataloader import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from network.Network import Network
from DataLoader.dataset.valtransform import ValTransform
from DataLoader.dataset.sources.cocosource import COCOSource
from DataLoader.dataset.testdataset import TestDataset
from new_evaluator.coco import coco
from utils import *
import matplotlib.pyplot as plt
from setting.read_setting import config as cfg
class TestExp():
    def __init__(self,
                 use_cuda,
                 data_dir,
                 save_dir,
                 use_tide):
        super(TestExp, self).__init__()
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.65
        self.test_size=(640,640)
        self.test_conf=0.001
        self.nmsthre=0.5
        self.use_cuda=use_cuda
        self.use_tide=use_tide
        self.data_dir=data_dir
        mode='test'
        self.source= COCOSource(data_dir=self.data_dir,mode=mode)
        self.test_dataset = TestDataset(base_dataset=self.source,
                                        preproc=ValTransform())
        self.loader = DataLoader(dataset=self.test_dataset,
                                 batch_size=1,
                                 use_cuda=self.use_cuda,
                                 use_shuffle=False,)
        self.num_img=len(self.source)
        self.act='silu'
        self.save_dir = self.get_index(save_dir)
        self.save_file= os.path.join(self.save_dir,'save_pred.json')
        self.save_txt= os.path.join(self.save_dir,'save_pred.txt')
        self.pic_type='.png'
        self.itr=0
        self.photo_dir=os.path.join(self.save_dir, "conf={},nms={}".format(self.test_conf,self.nmsthre))
        self.coco=coco(save_dir=save_dir,image_set=mode,year='2007')
        self.outcome_file=os.path.join(self.save_dir,"outcome.txt")
        self.names=[]
        self.values=[]
        os.makedirs(self.photo_dir, exist_ok=True)
    def get_index(self,save_dir):
        max = 0
        for dir in os.listdir(save_dir):
            if dir.isdigit():
                dir = int(dir)
                if (max < dir):
                    max = dir
        max = max + 1
        dir=os.path.join(save_dir,str(max))
        if not os.path.isdir(dir):
            os.mkdir(dir)
        return dir
    def load_yolox(self,pth_path=None):
        if self.use_cuda:
            self.model = Network('yolox_s').cuda()
            model = torch.load(pth_path, map_location=torch.device('cuda'))
        else:
            self.model=Network('yolox_s')
            model = torch.load(pth_path, map_location=torch.device('cpu'))
        torch.save(model,os.path.join(self.save_dir,'save_weight.pth'))
        self.model.eval()
        self.model.load_state_dict(model['model'], strict=False)
    def model_predict(self, image):
        with torch.no_grad():
            outputs = self.model(image, False)
            outputs[..., 4] = outputs[..., 4].sigmoid()
            outputs[..., 5] = outputs[..., 5].sigmoid()
            outputs[..., 6] = outputs[..., 6].sigmoid()
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.test_conf,
                self.nmsthre,
                class_agnostic=True)
            bboxes = []
            scores = []
            for output in outputs:
                if output is None:
                    return np.array([]), np.array([])
                bboxes.append(output[..., 0:4])
                scores.append(output[..., 4] * output[..., 5] * output[..., 6])
        return bboxes, scores
    def show_prediction(self,imgs,outcomes,targets,scores):
        for img_g,outcome,target,score in zip(imgs,outcomes,targets,scores):
            img_g=img_g.permute(1,2,0).cpu().numpy().astype(np.uint8)
            img_g=np.ascontiguousarray(img_g)
            pred_boxes=outcome[:,:4]
            gt_boxes=target[:,1:]
            img=img_g[...,2:3]
            img=np.repeat(img,3,axis=2)
            pos_mask=score>0.5
            img=self.draw_box(img,pred_boxes[pos_mask],(255,0,0),scores=score[pos_mask])
            img=self.draw_box(img,gt_boxes,(0,255,0))
            cv2.imwrite(os.path.join(self.photo_dir,"{}.png".format(self.itr)),img)
            self.itr=self.itr+1
    def save_pred(self):
        all_boxes =[[]]
        f = open(self.save_txt, 'w')
        ######################################
        for i,(imgs,_,_,targets,names) in enumerate(tqdm(self.loader)):
            outcomes, scores = self.model_predict(imgs)
            self.show_prediction(imgs,outcomes,targets,scores)
            if len(scores)<=0:
                all_boxes[0].append(np.array([[0,0,0,0,0]]))
            for name,score,outcome in zip(names,scores,outcomes):
                score=score.detach().cpu()
                outcome=outcome.detach().cpu()
                score=torch.reshape(score,[-1,1])
                boxes=outcome
                det=np.concatenate([score,boxes],-1)
                all_boxes[0].append(det)
                for s, o in zip(score, boxes):
                    f.write("{} {} {} {} {} {}\n".format(name,float(s), o[0], o[1], o[2], o[3]))
        self.coco._write_coco_results_file(all_boxes=all_boxes,res_file=self.save_file)
        return self.save_dir
    def tranform_int(self, boxes):
        box_list = []
        for box in boxes:
            box_list.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
        return box_list
    def draw_box(self,image,boxes_g,color=(0,255,0),class_name=None,scores=None):
        boxes = self.tranform_int(boxes_g).copy()
        for i,box in enumerate(boxes):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color)
            if scores is None:
                continue
            score=scores[i]
            text = '{}:{:.1f}%'.format(class_name, score * 100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(
                image,
                (box[0], box[1] + 31),
                (box[0] + txt_size[0] + 1, box[1] + int(1.5 * txt_size[1])+30),
                (255,0,0),
                -1)
            cv2.putText(image, text, (box[0], box[1] + txt_size[1]+30), font, 0.4, (255,255,255))
        return image
    def get_p_r(self,precision):
        plt.plot(np.arange(0., 1.01, 0.01), precision)
        plt.xlim(0, 1.)
        plt.ylim(0., 1.01)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(os.path.join(self.save_dir, 'p-r.png'))
    def read_coco_outcome(self):
        stats, precision = self.coco._do_detection_eval(res_file=self.save_file, output_dir=self.save_dir)
        names=[]
        values=[]
        for stat in stats:
            names.append(stat[0])
            values.append(stat[1])
        self.get_p_r(precision)
        np.save(os.path.join(self.save_dir,"precisions"),precision)
        self.write_excel(sheet_name="coco",names=names,values=values)
    def write_excel(self,sheet_name,names,values):
        sheet=self.f.add_sheet(sheet_name,True)
        for i,(name,value) in enumerate(zip(names,values)):
            sheet.write(0,i,name)
            sheet.write(1,i,value)
    def compute_ap(self):
        import xlwt
        self.f=xlwt.Workbook('encoding =utf-8')
        self.read_coco_outcome()
        self.f.save(os.path.join(self.save_dir, 'outcome.xls'))
if __name__=="__main__":
    # print(os.listdir(r"../datasets/ISDD/VOC2007"))
    exp=TestExp(
        use_cuda=cfg.use_cuda,
        data_dir=r'../datasets/SII',
        save_dir=r'../save_outcome',
        use_tide=True)
    exp.load_yolox(r'./best_ckpt.pth')
    exp.save_pred()
    exp.compute_ap()