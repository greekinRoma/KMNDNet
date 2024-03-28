import torch
import pandas as pd
import os
from easydict import EasyDict
def read_excel(loc,sheet_name):
    excel=pd.read_excel(loc,sheet_name=sheet_name)
    if excel.empty:
       raise 'trains are finished'
    set_pd=excel.head(1)
    set_dict=pd.DataFrame.to_dict(set_pd)
    set_dict=transform_type(set_dict)
    return set_dict
def begin_excel(loc,sheet_name):
    excel = pd.read_excel(loc,sheet_name=sheet_name)
    assert excel.empty!=True ,'Input is Empty!!!!!!!!!!!!!'
    dir_name=os.path.dirname(loc)
    writer=pd.ExcelWriter(os.path.join(dir_name,'save.xlsx'))
    excel.to_excel(writer,'save',index=False)
    writer.save()
    writer.close()
def finish_excel(loc,sheet_name):
    #获得路
    dir_name = os.path.dirname(loc)
    #
    excel = pd.read_excel(loc, sheet_name=sheet_name)
    finishes_excel=pd.read_excel(os.path.join(dir_name,'finish.xlsx'))
    content=excel.head(1)
    finishes_excel=finishes_excel.append(content)
    writer=pd.ExcelWriter(os.path.join(dir_name,'finish.xlsx'))
    finishes_excel.to_excel(writer, 'finish', index=False)
    writer.save()
    writer.close()
    #
    excel = excel.drop(0)
    writer = pd.ExcelWriter(os.path.join(loc))
    excel.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()
    writer.close()
def transform_type(datas):
    for key in datas.keys():
        if key=='mosaic_scale' or key=='mixup_scale':
            for i in datas[key].keys():
                tmp_data=datas[key][i]
                split=tmp_data.split('t')
                datas[key][i]=(float(split[0]),float(split[1]))
    return datas
def generate_args(main_dir:str,set_dict:dict,is_read_excel:bool):
    if is_read_excel:
        assert len(set_dict)>0,'names and setting is empty!!!!please check the path of loc!!!'
    args=EasyDict()
    #----------------------------NetWork----------------------------#
    args['net_name'] = 'yolox_s'
    args['name']=0
    args["mode"]='ssd'
    args["assign_mode"]='simota'
    #----------------------------Data----------------------------#
    #dataloader
    args['coco_data_dir'] =os.path.join(main_dir,r'datasets/SII')
    args['target_dir'] = os.path.join(main_dir,r'datasets')
    args['use_shuffle'] = True
    args['cache_type'] = "ram"
    args['cache'] = True
    args['batch_size'] = 32
    #train_dataset
    args['maxtarget'] = 3
    args['aug_prob'] = 1.
    args['mixup_mosaic_prob'] = 0.5
    args['mixup_prob'] = 0.5
    args['mosaic_prob'] = 0.5
    args['flip_prob'] = 0.5
    args['gen_prob'] = 0.5
    args['img_size'] = (640, 640)
    args['enable_mosaic'] = True
    args['degrees'] = 10.0
    args['translate'] = 0.1
    args['mosaic_scale'] = (0.1, 2.)
    args['mixup_scale'] = (0.5, 1.5)
    args['shear'] = 2.0
    args['enable_mixup'] = True
    #training setting
    args['use_valid']=False
    args['root_dir']=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    args['image_ext'] = '.png'
    args['use_cuda'] =True
    args['max_epoch']=300
    args['input_size']=640
    args['strides']=[8]
    args['in_channels']=[256]
    args['fp16']=False
    args['data_type']=torch.float32
    for name,setting in set_dict.items():
        assert name in args.keys(),'{} can not be found! keys error! Please reset the excel!!!'.format(name)
        args[name]=setting[0]
    return args
try:
    main_dir=r'../'
    set_dict=read_excel(os.path.join(main_dir,'input.xlsx'),'input')
except:
    main_dir = r'./'
    set_dict = read_excel(os.path.join(main_dir, 'input.xlsx'), 'input')
config=generate_args(main_dir,set_dict,True)