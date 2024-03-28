from DataLoader.dataset.dataaugment.mixup import *
from DataLoader.dataset.dataaugment.mosaic import *
class AugmentController():
    def __init__(self,
                 input_w: int,
                 input_h: int,
                 degrees: float = 10.0,
                 translate: float = 0.1,
                 mosaic_scale= (0.1, 2.),
                 mixup_scale= (0.5, 1.5),
                 shear: float = 2.0,
                 ):
        self.input_w=input_w
        self.input_h=input_h
        self.degrees=degrees
        self.translate=translate
        self.mosaic_scale=mosaic_scale
        self.mixup_scale=mixup_scale
        self.shear=shear
    def get_mosaic(self):
        return   mosaic(
                        degrees=self.degrees,
                        translate=self.translate,
                        mosaic_scale=self.mosaic_scale,
                        shear=self.shear,
                        input_w=self.input_w,
                        input_h=self.input_h
        )
    def get_mixup(self):
        return mixup(
            input_w=self.input_w,
            input_h=self.input_h,
            mixup_scale=self.mixup_scale
        )