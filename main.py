import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/github/metatrans')
import shutil
import traceback
import os
from torch.utils.data import  DataLoader
import time

# from common.originize_df import originze_df_maml_four
# from common.lymph.originize_ly_df import ly_originze_df,originze_source_data_pre
# from common.lymph.dataloader_ly import LY_MetaDataset, LY_dataset
from common.data_enter import oct_data, source_data_pre,lymph_data
from common.imagenet.originize_imagenet import originze_imagenet

from common.build_tasks import Train_task,Meta_Test_task
from common.dataloader import *
import common.utils as utils
from  imaml.imaml_enter import imaml_enter
from maml.maml_enter import maml_enter
from reptile.reptile_enter import reptile_enter
from  dl.dl_enter import dl_enter, predict_enter
from data.pre_train.zk_data_pro import zk_data
from data.pre_train.st_data_pro import st_sub_data

import run as dl_exe

def main(args):

    if args.datatype == "OCT":
        test_data_ls, train_data_ls = oct_data()
    elif args.datatype == 'zk':
        test_data_ls, train_data_ls = zk_data()
    elif args.datatype == 'st_sub_pretrain':
        test_data_ls, train_data_ls = st_sub_data()
    elif args.datatype == 'imagenet':
        test_data_ls, train_data_ls = originze_imagenet()
    elif args.datatype == 'source_data_preTran':
        test_data_ls, train_data_ls = source_data_pre()
    elif 'lymph' in args.datatype or 'thyroid' in args.datatype or 'ffpe' in args.datatype or 'bf' in args.datatype or 'multi_centers' in args.datatype or 'camely' in args.datatype or 'crc' in args.datatype or 'background' in args.datatype:
        test_data_ls, train_data_ls = lymph_data(args)
    
    

    if args.alg == 'dl':
        dl_enter(args, test_data_ls)
    elif args.alg=='maml':
        maml_enter(args, train_data_ls, test_data_ls)
    elif args.alg=='reptile':
        reptile_enter(args, train_data_ls, test_data_ls)
    elif args.alg=='Neumann':
        pass
    elif args.alg=='CAVIA':
        pass
    elif args.alg=='imaml':
        imaml_enter(args, train_data_ls, test_data_ls)
    elif args.alg=='FOMAML':
        pass
    elif 'predict' in args.alg:
        predict_enter(args, test_data_ls, args.meta_epoch_for_predict)


    else:
        raise ValueError('Not implemented Meta-Learning Algorithm')
    
    print('over+++++++++++++++++++')



def parse_args():
    import argparse

    parser = argparse.ArgumentParser('Gradient-Based Meta-Learning Algorithms')
    
    # base settings
    parser.add_argument('--is_run_command', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--is_debug', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--is_remove_exres', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--description_name', type=str, default='description')
    parser.add_argument('--description', type=str, default='hh')
    parser.add_argument('--datatype', type=str, default='OCT')
    parser.add_argument('--data_path_prefix', type=str, default='')
    parser.add_argument('--seed', type=int, default=1) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='0 = GPU.')
    parser.add_argument('--save_path', type=str, default='tmp')
    parser.add_argument('--num_classes', type=int, default=4)# Total number of fc out_class
    parser.add_argument('--prefix', type=str, default='debug') # The network architecture 
    parser.add_argument('--acc_average', type=str, default='weighted') # The network architecture 

    # dataset
    # camely17
    parser.add_argument('--predict_class', type=str, nargs='+', default=[])
    
    # multi-centers
    parser.add_argument('--multi_centers_test', type=str, default='') # jingxia throid crc or none

    # net 
    parser.add_argument('--is_load_imagenet', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--is_load_zk', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--is_load_imagenet_zk', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--is_load_st_sub', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--is_meta_load_imagenet', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--is_lock_notmeta', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--load',type=str, default="")
    # network settings
    parser.add_argument('--net', type=str, default='resnet18')
    # parser.add_argument('--n_conv', type=int, default=4)
    # parser.add_argument('--n_dense', type=int, default=0)
    # parser.add_argument('--hidden_dim', type=int, default=64)
    # parser.add_argument('--in_channels', type=int, default=1)
    # parser.add_argument('--hidden_channels', type=int, default=64,
    #     help='Number of channels for each convolutional layer (default: 64).')
    # transformer 深度
    parser.add_argument('--trans_depth', type=int, default=12)

    # for predict
    parser.add_argument('--meta_epoch_for_predict', type=int, default=0)

    parser.add_argument('--load_interrupt_path', type=str, default="", help='Load model from a .pth file for interrupt recover')
    parser.add_argument('--best_acc', type=float, default=0.0)
    parser.add_argument('--best_epoch', type=int, default=0)

    # for dl/
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--is_save_val_net', type=lambda x: (str(x).lower() == 'true'), default=False) # 是否保存验证集上最好的模型
    parser.add_argument('--batch_size_train', type=int, default=16)
    parser.add_argument('--batch_size_val', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading (default: 4).')

    # for predict
    parser.add_argument('--isheatmap', type=lambda x: (str(x).lower() == 'true'), default=False)

    # algorithm settings
    parser.add_argument('--alg', type=str, default='iMAML')
    parser.add_argument('--n_inner', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=1e-2)
    parser.add_argument('--inner_opt', type=str, default='SGD')
    parser.add_argument('--outer_lr', type=float, default=1e-3)
    parser.add_argument('--outer_opt', type=str, default='Adam')
    parser.add_argument('--lr_sched', type=lambda x: (str(x).lower() == 'true'), default=False)

    parser.add_argument('--n_train_tasks', type=int, default=1000)# Total number of trainng tasks
    parser.add_argument('--n_val_tasks', type=int, default=250)# Total number of trainng tasks
    parser.add_argument('--n_test_tasks', type=int, default=1)# Total number of testing tasks

    # meta training settings
    parser.add_argument('--n_meta_epoch', type=int, help='',default=50)
    parser.add_argument('--n_way', type=int, help='n way', default=2)
    parser.add_argument('--k_shot', type=int, help='k shot for support set', default=5)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    parser.add_argument('--meta_size', type=int, help='meta batch size, namely meta_size',default=4)

    # test tasks
    # 必须大于15，保证每个task,包含common所有类
    parser.add_argument('--test_k_shot', type=int, help='k shot for support set', default=5)
    parser.add_argument('--test_k_qry', type=int, help='k shot for query set', default=5)
    # 默认值为1，最好不要改动
    parser.add_argument('--test_meta_size', type=int, help='meta batch size, namely meta_size',default=1)


    
    # imaml specific settings
    parser.add_argument('--lambda', type=float, default=2.0)# 并没有使用到啊擦
    parser.add_argument('--version', type=str, default='GD')
    parser.add_argument('--cg_steps', type=int, default=5) 
    
    

    args = parser.parse_args()

    if not args.is_run_command:
        args_dict = dl_exe.gen_args() 
        for key, value in args_dict.items():
            setattr(args, key, value)
    if args.is_debug:
        setattr(args, 'save_path', 'tmp')
        setattr(args, 'n_epoch', 2)
        setattr(args, 'n_meta_epoch', 2)
        setattr(args, 'n_train_tasks', 10)
        setattr(args, 'n_val_tasks', 2)
    return args

def create_store_dir(args):
    args.project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/github/metatrans'
    
    begin_time = time.time()
    time_name=time.strftime('%Y-%m-%d-%H-%M-%S')

    args.store_dir =str(os.path.join(args.project_path, args.save_path, args.datatype, args.alg.lower(), args.net, time_name))
    utils.mkdir(args.store_dir)

    # 创建一个说明文件
    description_file = os.path.join(args.store_dir,  args.description_name)
    with open(str(description_file), 'w') as f:
        f.write(args.description)

if __name__ == '__main__':
    args = parse_args()
    
    # utils.set_gpu(args.gpu)
    torch.cuda.set_device(args.gpu[0])
    utils.set_seed(args.seed)

    create_store_dir(args)
    utils.save_args_to_file(args, os.path.join(args.store_dir, 'args.json'))

    try:
        main(args)
    except Exception:
        print(traceback.print_exc())
        if args.is_remove_exres:
            shutil.rmtree(args.store_dir)
