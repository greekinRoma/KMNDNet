import os
import cv2
class TargetSource():
    def __init__(self,
                 data_dir):
        self.data_dir=os.path.join(data_dir,'targets')
        self.ids=[name for name in os.listdir(self.data_dir)]
        self.num_imgs=len(self.ids)
        self.name='target'
        self.path_filename = [os.path.join(self.data_dir, ids) for ids in self.ids]
    def __len__(self):
        return len(self.ids)
    def read_img(self, idx):
        img_path = self.path_filename[idx]
        img = cv2.imread(img_path)
        return img
    def __getitem__(self, index):
        img=self.read_img(index)
        h,w,_=img.shape
        return img,w,h
if __name__=='__main__':
    target_source=TargetSource(data_dir=r'../../../../datasets')
    for i in range(len(target_source)):
        img,w,h=target_source[i]
        cv2.rectangle(img,(int(w//4),int(h//4)),(int(w*3//4),int(h*3//4)),color=(0,255,255))
        cv2.imshow('outcome', img)
        cv2.waitKey(0)