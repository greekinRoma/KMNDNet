import numpy as np
class TestDataset():
    def __init__(self,
                 base_dataset,
                 preproc
                 ):
        super().__init__()
        self.base_dataset=base_dataset
        self.preproc=preproc
        self.img_size=self.base_dataset._input_dim()
    def __len__(self):
        return len(self.base_dataset)
    def _input_dim(self):
        return self.img_size
    def __getitem__(self,
                    index:int):
        img,target,name, img_info, img_id = self.base_dataset[index]
        img,_=self.preproc(img, target, self.img_size)
        return img,np.ones_like(img),0,target, name, img_info, img_id
    def reset_pro(self):
        pass