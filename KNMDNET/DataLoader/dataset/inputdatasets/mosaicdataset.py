import random
import numpy as np
from DataLoader.dataset.dataaugment.mosaic import mosaic
from DataLoader.dataset.dataaugment.mixup import mixup
class MosaicDataset():
    def __init__(self,
                 mixup_prob:float,
                 mosaic_prob:float,
                 mosaic:mosaic,
                 mixup:mixup,
                 mask_dataset,
                 base_dataset,
                 preproc,
                 ):
        super().__init__()
        self.mixup_prob=mixup_prob
        self.mosaic_prob=mosaic_prob
        self.base_dataset=base_dataset
        self.mask_dataset=mask_dataset
        self.mosaic=mosaic
        self.mixup=mixup
        self.preproc=preproc
        self.img_size=self.base_dataset._input_dim()
    def __len__(self):
        return len(self.base_dataset)
    def _input_dim(self):
        return self.img_size
    def __getitem__(self,
                    index:int):
        img,target,name, img_info, img_id = self.base_dataset[index]

        mask, name=self.mask_dataset[index]
        img=np.ascontiguousarray(img)
        if random.random()<self.mosaic_prob:
            img,mask,target=self.mosaic(dataset=self.base_dataset,mask_dataset=self.mask_dataset,mask=mask,inp=img,label=target)
        if random.random()<self.mixup_prob:
            img,mask,target=self.mixup(dataset=self.base_dataset,mask_dataset=self.mask_dataset,origin_img=img,origin_mask=mask,origin_labels=target)
        return img,mask,target, name, img_info, img_id