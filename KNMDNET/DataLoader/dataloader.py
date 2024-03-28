import numpy as np
import uuid
import torch
class DataLoader():
    def __init__(self,
                 dataset,
                 batch_size:int,
                 use_shuffle:bool=True,
                 use_cuda:bool=True):
        self.dataset=dataset
        self.num_img=len(self.dataset)
        self.batch_size=batch_size
        self.use_shuffle = use_shuffle
        self.use_cuda=use_cuda
        if self.use_shuffle:
            seed = int(uuid.uuid4())
            np.random.seed(seed % 1000)
        self.resetmap()
    def __len__(self):
        return self.num_batch
    def split(self,map):
        num_img=len(map)
        list_m=[]
        i=0
        while(True):
            temp_map = []
            for j in range(self.batch_size):
                temp_map.append(map[i])
                i=i+1
                if (i>num_img-1):
                    list_m.append(temp_map)
                    return list_m,len(list_m)
            list_m.append(temp_map)
    def resetmap(self):
        map=np.arange(len(self.dataset))
        if self.use_shuffle:
            np.random.shuffle(map)
        self.map,self.num_batch=self.split(map)
        self.num_img = len(self.dataset)
    def reset_prob(self,prob):
        self.dataset.reset_prob(prob)
    def close_mosaic(self):
        self.dataset.enable_mosaic=False
    def open_mosaic(self):
        self.dataset.enable_mosaic=True
    def close_origin(self):
        self.dataset.enable_origin=False
    def open_origin(self):
        self.dataset.enable_origin=True
    def __next__(self):
        targets = []
        names = []
        imgs = []
        masks=[]
        use_augs=[]
        if (self.itr > self.num_batch-1):
            raise StopIteration
        for idx in self.map[self.itr]:
            img,mask,use_aug,target, name, _, _ = self.dataset[idx]
            use_augs.append(use_aug)
            masks.append(mask)
            imgs.append(img)
            names.append(name)
            targets.append(target)
        self.itr = self.itr + 1
        imgs = np.array(imgs)
        targets=np.array(targets)
        masks=np.array(masks)
        use_augs=np.array(use_augs)
        if self.use_cuda:
            imgs = torch.from_numpy(imgs).cuda()
            targets=torch.from_numpy(targets).cuda()
            masks=torch.from_numpy(masks).cuda()
            use_augs=torch.from_numpy(use_augs).cuda()
        else:
            imgs = torch.from_numpy(imgs)
            targets = torch.from_numpy(targets)
            masks=torch.from_numpy(masks)
            use_augs = torch.from_numpy(use_augs)
        return imgs,masks,use_augs,targets,names
    def __iter__(self):
        self.itr=0
        return self