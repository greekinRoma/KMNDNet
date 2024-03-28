from trainer import Trainer
from exp.exp import MyExp
from new_evaluator.test_exp import TestExp
import traceback
import numpy as np
from setting.read_setting import *
import gc
import os
def get_save_path(set_dict:dict,save_dir:str):
    save_dir_name=''
    for name,value in set_dict.items():
        save_dir_name=save_dir_name+'{}={}_'.format(name,value[0])
    save_path=os.path.join(save_dir,save_dir_name)
    os.makedirs(save_path,exist_ok=True)
    return save_path
def launch():
    set_dict = read_excel(r'./input.xlsx', 'input')
    args = generate_args('./',set_dict,is_read_excel=True)
    save_pth = get_save_path(set_dict, save_dir=r'save_outcome')
    save_setting(args=args, save_path=save_pth)
    exp = MyExp(args=args, save_path=save_pth)
    trainer = Trainer(exp)
    trainer.train()
    test_exp = TestExp(data_dir=args['coco_data_dir'], save_dir=save_pth, use_tide=False,use_cuda=torch.cuda.is_available())
    test_exp.load_yolox('./training_save/yolox_s_OTA/best_ckpt.pth')
    save_dir=test_exp.save_pred()
    exp.save_index(save_dir)
    test_exp.compute_ap()
    finish_excel(r'./input.xlsx','input')
    del test_exp, trainer, exp
    gc.collect()
    os.system('cls')
def save_setting(args:dict,save_path:str):
    save_file_path = os.path.join(save_path, 'save.txt')
    f = open(save_file_path, 'w')
    for name, value in args.items():
        f.write(f'{name}:{value}\n')
    f.close()
if __name__=='__main__':
    try:
        begin_excel(r'input.xlsx', 'input')
        del begin_excel
        gc.collect()
        while (True):
            launch()
    except (Exception, BaseException) as e:
        traceback.print_exc()
        print('traceback.format_exc():\n%s' % traceback.format_exc())
        raise