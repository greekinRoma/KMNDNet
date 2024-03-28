import datetime
import random
import time
import loguru
import numpy as np
from DataLoader.dataprefetcher import DataPrefetcher
from exp.exp import MyExp
from utils import *
from utils.model_utils import *
import traceback
import torch
import os
from loguru import logger
import shutil
class Trainer:
    def __init__(self, exp:MyExp):
        self.exp = exp
        self.max_epoch = int(exp.max_epoch)
        self.amp_training = exp.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=exp.fp16)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.use_cuda=self.exp.use_cuda
        # data/dataloader related attr
        self.data_type = torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0
        self.start_epoch=0
        #controller
        self.use_valid=exp.use_valid
        # metric record
        self.file_name =exp.file_name
        self.loss_f=self.exp.get_loss_f()
        self.save_path=exp.save_path
        self.txt_writer=exp.txt_writer
        os.makedirs(self.file_name,exist_ok=True)
        setup_logger(self.file_name,distributed_rank=0,filename="train_log.txt",mode="a")
    def train(self):
        try:
            self.before_train()
            self.train_in_epoch()
        except (Exception,BaseException) as e:
            traceback.print_exc()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            raise
        finally:
            self.after_train()
    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()
            if self.epoch>160:
                break
    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()
    def train_one_iter(self):
        iter_start_time = time.time()
        inps,masks,use_augs,targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        data_end_time = time.time()
        with torch.cuda.amp.autocast(enabled=self.amp_training):
            image=inps[0].detach().cpu().numpy()
            # import cv2
            # print(np.max(image[0]))
            # print(np.max(image[1]))
            # cv2.imshow('image0',image[0]/255.)
            # cv2.imshow('image1',image[1]/255.)
            # cv2.imshow('image2',image[2]/255.)
            # cv2.waitKey(0)
            preds, grids, strides,regs = self.model(inps,training=True)
            outputs,loss_list=self.loss_f(
                        use_augs=use_augs,
                        masks=masks,
                        targets=targets,
                         strides=strides,
                         grids=grids,
                         outputs=preds,
                         regs=regs)
        loss = outputs['total_loss']
        try:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        except:
            logger.info(f'Total Loss is {loss}')
        if self.use_model_ema:
            self.ema_model.update(self.model)
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for k,param_group in enumerate(self.optimizer.param_groups):
            if k==3:
                param_group["lr"] = lr
            else:
                param_group["lr"] = lr

        '''
        是否对网络输出图像进行测试
        from VisionTools.show_img import vision_outcome
        vision_outcome(name='outcome',imgs=inps,labels=targets)
        '''
        iter_end_time = time.time()
        self.meter.update(
            iter_time = iter_end_time - iter_start_time,
            data_time = data_end_time - iter_start_time,
            lr = lr,
            ** outputs,)
    def test_dataloader(self):
        from VisionTools.save_img import save_outcome
        self.test_trainloader_dir = os.path.join(self.file_name, r'trainloader')
        self.test_testloader_dir = os.path.join(self.file_name, r'testloader')
        if os.path.exists(self.test_trainloader_dir):
            shutil.rmtree(self.test_testloader_dir)
        if os.path.exists(self.test_trainloader_dir):
            shutil.rmtree(self.test_trainloader_dir)
        os.makedirs(self.test_trainloader_dir, exist_ok=True)
        os.makedirs(self.test_testloader_dir, exist_ok=True)
        for imgs,_,_,targets,names in self.train_loader:
            save_outcome(names=names,
                         save_dir=self.test_trainloader_dir,
                         labels=targets,
                         imgs=imgs,
                         need_change=True)
        test_loader=self.exp.get_testloader()
        for imgs,_,_,targets,names in test_loader:
            save_outcome(names=names,
                         save_dir=self.test_testloader_dir,
                         labels=targets,
                         imgs=imgs,
                         need_change=False)
    def before_train(self):
        self.optimizer = self.exp.get_optimizer(self.exp.batch_size)
        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_trainloader()
        logger.info("init prefetcher, this might take one minute or less...")
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)
        self.lr_scheduler = self.exp.get_lr_scheduler(self.exp.basic_lr_per_img * self.exp.batch_size, self.max_iter)
        model=self.exp.get_model()
        try:
            if self.use_cuda:
                net=torch.load('./yolox_s.pth', map_location=torch.device('cuda'))["model"]
            else:
                net = torch.load('yolox_s.pth', map_location=torch.device('cpu'))["model"]
            model=load_ckpt(model,net)
        except (Exception,BaseException) as e:
            traceback.print_exc()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch
        self.model = model
        self.test_evaluator = self.exp.get_test_evaluator()
        #self.test_dataloader()
        np.random.seed(int(time.time()))
    def after_train(self):
        logger.info("Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100))
    def before_epoch(self):
        '''是否最后停止数据增强开关
        if self.epoch>=self.max_epoch - self.exp.no_aug_epochs:
            self.train_loader.close_mosaic()
            self.no_aug=True
            logger.info('Stop Data Augment Now!!!!!!!!!!')
        '''
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        self.exp.reset_map()
        prob=1.-(self.epoch/self.max_epoch)**2
        self.train_loader.reset_prob(prob)
        self.prefetcher = DataPrefetcher(self.train_loader,
                                         use_cuda=self.use_cuda)
        self.prefetcher.preload()
    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")
        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()
    def before_iter(self):
        t=time.time()
        torch.manual_seed(int(t))

    def after_iter(self):
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

            logger.info(
                "{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size, eta_str))
            )
            self.meter.clear_meters()

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

#/--------------------------------------------------test evaluator-------------------------------------------------------/#
    def evaluate_test_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module
        with adjust_status(evalmodel, training=False):
            self.test_evaluator.push_model(evalmodel.eval())
            stats= self.test_evaluator.eval()
        return stats
    def save_test_model(self,update_best_ckpt,ap50_95):
        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)
    def evaluate_and_save_model(self):
        stats=self.evaluate_test_model()
        ap50=stats[1][1]
        print(ap50)
        # ap50_95=stats[0][1][1]
        # logger.info("AP@50:{}".format(ap50))
        # logger.info("AP@50:95:{}".format(ap50_95))
        update_best_ckpt = ap50 > self.best_ap
        self.best_ap = max(self.best_ap, ap50)
        self.save_test_model(update_best_ckpt=update_best_ckpt,ap50_95=ap50)
    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        save_model = self.ema_model.ema if self.use_model_ema else self.model
        logger.info("Save weights to {}".format(self.file_name))
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_ap": self.best_ap,
            "curr_ap": ap,
        }
        save_checkpoint(
            ckpt_state,
            update_best_ckpt,
            self.file_name,
            ckpt_name,
        )
