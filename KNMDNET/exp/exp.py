from network.Network import Network
from DataLoader.dataloader import DataLoader
from torch import nn
from utils import LRScheduler
import torch
import os
from losses.loss import Loss_Fuction
from new_evaluator.train_eval import Evaluator
from utils.Txt_controller import txt_writer
from DataLoader.dataset.sources.cocosource import COCOSource
from DataLoader.dataset.traindatasets import TrainDatasets
from DataLoader.dataset.testdataset import TestDataset
from DataLoader.dataset.traintransform import TrainTransform
from DataLoader.dataset.valtransform import ValTransform
from DataLoader.dataset.dataaugment import AugmentController
from DataLoader.dataset.sources.masksource import MaskSource
class MyExp(nn.Module):
    def __init__(self,args,save_path):
        super(MyExp,self).__init__()
        self.data_dir =args['coco_data_dir']
        self.target_dir=args['target_dir']
        self.use_cuda=args['use_cuda']
        #------------------------dataset---------------------------#
        self.enable_mosaic=args['enable_mosaic']
        self.use_valid=args['use_valid']
        self.name=args['name']
        self.batch_size = args['batch_size']
        self.mosaic_prob = args['mosaic_prob']
        self.mixup_prob = args['mixup_prob']
        # ------------------------------------------------------------------------dataset_controller-------------------------------------------------------------#
        self.trainval_source = COCOSource(data_dir=self.data_dir,
                                    mode='trainval',
                                    cache_type=args['cache_type'],
                                    cache=args['cache'])
        self.test_source=COCOSource(data_dir=self.data_dir,
                                    mode='test',
                                    cache_type=args['cache_type'],
                                    cache=args['cache'])
        self.mask_source=MaskSource(ids=self.trainval_source.send_ids(),
                                    data_dir=self.data_dir,
                                    cache_type=args['cache_type'],
                                    cache=args['cache'])
        #self.mask_source.get_ids()
        self.aug_controller = AugmentController(input_w=args['img_size'][0],
                                           input_h=args['img_size'][1],
                                           degrees=args['degrees'],
                                           translate=args['translate'],
                                           mosaic_scale=args['mosaic_scale'],
                                           mixup_scale=args['mixup_scale'],
                                           shear=args['shear'])
        self.train_dataset = TrainDatasets(
            masksource=self.mask_source,
            vocsource=self.trainval_source,
            targetsource=None,
            maxtarget=args['maxtarget'],
            aug_prob=args['aug_prob'],
            mixup_mosaic_prob=args['mixup_mosaic_prob'],
            mixup_prob=args['mixup_prob'],
            mosaic_prob=args['mosaic_prob'],
            flip_prob=args['flip_prob'],
            gen_prob=args['gen_prob'],
            mixup=self.aug_controller.get_mixup(),
            mosaic=self.aug_controller.get_mosaic(),
            preproc=TrainTransform()
        )
        self.test_dataset = TestDataset(base_dataset=self.test_source,
                                        preproc=ValTransform())
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.batch_size,
                                       use_shuffle=True,
                                       use_cuda=self.use_cuda)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=1,
                                      use_shuffle=False,
                                      use_cuda=self.use_cuda)
        #---------------------------path----------------------------#
        self.save_path = save_path
        self.file_name = os.path.join('training_save', args['net_name'] + '_' + "OTA")
        #---------------------------evaluator-----------------------#
        self.test_evaluator = Evaluator(self.test_loader,
                                        need_change=False)
        self.txt_writer=txt_writer(self.file_name,r'preform.txt')
        # ----------------------------------------------------------#
        self.num_batch=len(self.train_loader)
        self.max_epoch=args['max_epoch']
        self.input_size=640
        self.fp16=False
        if self.use_cuda:
            self.model=Network(args['net_name']).cuda().train()
        else:
            self.model=Network(args['net_name']).train()
        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 1
        # max training epoch
        self.max_epoch = args['max_epoch']
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 15
        # apply EMA during training
        self.ema = True
        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 2
        # eval period in epoch, for example,use_valid
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 1
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
    def get_lr_scheduler(self, lr, iters_per_epoch):
        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio)
        return scheduler
    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size
            pg0, pg1, pg2,pg3 = [], [], [],[]  # optimizer parameter groups
            for k,v in self.model.named_parameters():
                if 'scale' in k:
                    pg3.append(v)
            for k, v in self.model.named_modules():
                # if 'lca' in k:
                #     if hasattr(v,"bias") and isinstance(v.bias, nn.Parameter):
                #         pg3.append(v.bias)
                #     if hasattr(v,"weight") and isinstance(v.weight, nn.Parameter):
                #         pg3.append(v.weight)
                #     elif isinstance(v,nn.BatchNorm2d) or "bn" in k:
                #         pg3.append(v.weight)
                #     continue
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay
            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params":pg2})
            optimizer.add_param_group({"params":pg3})
            self.optimizer = optimizer
        return self.optimizer
    def get_trainloader(self):
        return self.train_loader
    def get_model(self):
        return self.model
    def get_loss_f(self,assignment_mode='OTA'):
        return Loss_Fuction(assignment_mode)
    def reset_map(self):
        self.train_loader.resetmap()
    def get_test_evaluator(self):
        return self.test_evaluator
    def get_controller(self):
        return self.controller
    def get_testloader(self):
        return self.test_loader
    def save_index(self,save_dir):
        self.txt_writer.save_file(save_dir)