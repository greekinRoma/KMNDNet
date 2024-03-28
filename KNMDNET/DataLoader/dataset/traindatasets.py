from DataLoader.dataset.inputdatasets.mosaicdataset import MosaicDataset
from DataLoader.dataset.sources.vocsource import VocSource
from DataLoader.dataset.sources.targetsource import TargetSource
from DataLoader.dataset.dataaugment.mixup import mixup
from DataLoader.dataset.dataaugment.mosaic import mosaic
from DataLoader.dataset.sources.masksource import MaskSource
import random
class TrainDatasets():
    def __init__(self,
                 vocsource:VocSource,
                 targetsource:TargetSource,
                 masksource:MaskSource,
                 maxtarget:int,
                 aug_prob:float,
                 mixup_mosaic_prob:float,
                 gen_prob:float,
                 flip_prob:float,
                 mosaic_prob:float,
                 mixup_prob:float,
                 mixup:mixup,
                 mosaic:mosaic,
                 preproc
                 ):
        self.masksource=masksource
        self.mixup_mosaic_prob=mixup_mosaic_prob
        self.gen_prob=gen_prob
        self.aug_prob=aug_prob
        self.vocsource=vocsource
        self.targetsource=targetsource
        self.preproc=preproc
        self.img_size = self.vocsource.img_size
        self.mosaic_datasets=MosaicDataset(mosaic=mosaic,
                                          mixup=mixup,
                                          mixup_prob=mixup_prob,
                                          mosaic_prob=mosaic_prob,
                                          mask_dataset=self.masksource,
                                          base_dataset=self.vocsource,
                                          preproc=preproc)
    def __len__(self):
        return len(self.vocsource)
    def _input_dim(self):
        return self.vocsource.img_size
    def pull_item(self,index:int):
        if random.random()<self.aug_prob:
            return self.mosaic_datasets[index],1
        else:
            img, target, name, img_info, img_id=self.vocsource[index]
            mask_img,_=self.masksource[index]
            return [img,mask_img,target, name, img_info, img_id],0
    def __getitem__(self, index:int):
        [img,mask_img,target, name, img_info, img_id],use_aug=self.pull_item(index)
        img, target = self.preproc(img, target, self.img_size)
        return img,mask_img,use_aug,target, name, img_info, img_id
    def reset_prob(self,prob):
        self.aug_prob=prob
if __name__=='__main__':
    from DataLoader.dataset.dataaugment import AugmentController
    from DataLoader.dataset.traintransform import TrainTransform
    from DataLoader.dataloader import DataLoader
    from VisionTools.show_img import vision_outcome
    import cv2
    import numpy as np
    voc_source = VocSource(data_dir=r'C:\Users\27227\Desktop\datasets\ISDD\data\VOCdevkit2007\VOC2007',mode='trainval')
    target_source = TargetSource(data_dir=r'C:\Users\27227\Desktop\datasets')
    mask_source=MaskSource(data_dir=r'C:\Users\27227\Desktop\datasets\ISDD\data\VOCdevkit2007\VOC2007',ids=voc_source.send_ids())
    aug_controller = AugmentController(input_w=640,
                                       input_h=640,
                                       degrees=10.0,
                                       translate=0.1,
                                       mosaic_scale=(0.5, 1.5),
                                       mixup_scale=(0.5, 1.5),
                                       shear=2.0, )
    train_datasets=TrainDatasets(
        masksource=mask_source,
        vocsource=voc_source,
        targetsource=target_source,
        maxtarget=10,
        aug_prob=1.0,
        mixup_mosaic_prob=0.5,
        mixup_prob=1.,
        mosaic_prob=1.,
        flip_prob=0.5,
        gen_prob=0.5,
        mixup=aug_controller.get_mixup(),
        mosaic=aug_controller.get_mosaic(),
        preproc=TrainTransform()
    )
    dataloader = DataLoader(dataset=train_datasets, batch_size=1, use_cuda=False)
    for imgs,masks,use_augs,targets,names in dataloader:
        pass
        #vision_outcome(name='outcome',labels=targets,imgs=imgs)
        #for mask,use_aug in zip(masks,use_augs):
        #    mask=np.array(mask)
        #    cv2.imshow("mask",mask*255.)
        #    cv2.waitKey(0)


