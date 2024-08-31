import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')


import os
import argparse
import time
import csv
import logging

from copy import deepcopy

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import torchvision.models

from common.dataloader import *
import common.utils as utils
from common.build_tasks import Train_task,Meta_Test_task
import common.originize_df as ori
from common.meta.meta_comm import meta_comm
import dl.dl_func as dlf


def main():

    metric_dir = os.path.join(store_dir, 'metric_' +  time_name + '.csv')
    # 创建
    with open(str(metric_dir), 'w') as f:
        fields = ['EPOCH','train_acc_last_task', 'val_acc_ave', 'val_recall_ave_pvrl', 'val_recall_ave_others']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)
        
    # 创建一个记录每一次梯度的，用于观察
    metric_dir_loop = os.path.join(store_dir, 'metric_' +  time_name + '_loop.csv')
    with open(str(metric_dir_loop), 'w') as f:
        fields = ['EPOCH'] + list(range(1, args.n_inner_updates+1, 1))
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)      
    

    # df_train, df_val, df_test_ls,train_classes, val_classes, test_classes = ori.originze_df(project_path)
    # _, _, df_test_ls,_, _, _ = ori.originze_df_sub(project_path) # 不用这个了

    target_classes = ['normal', 'PIC', 'PVRL',  'RP'] 
    _, df_test_ls = ori.originze_df_maml_four(project_path, target_classes)

    test_shot = args.test_k_shot  # ，每个测试任务，每个类别包含的训练样本的数量
    test_ways = args.n_way  # 
    test_query = args.test_k_qry  # ， 每个测试任务，每个类别包含的测试样本的数量
    
    # support相当于验证集，可用于算法本身的优化和调整，但不能作为最终模型好坏的评价数据
    test_query_fileroots_alltask, test_support_fileroots_all_task = [], []
    
    final_test_alltask = []#用于最终测试

    for each_task in range(args.n_test_tasks):  # 测试任务总数
        test_task = Meta_Test_task(test_shot, test_query, df_test_ls)
        test_query_fileroots_alltask.append(test_task.query_roots)
        test_support_fileroots_all_task.append(test_task.support_roots)
        final_test_alltask.append(test_task.test_roots)
    
    
    dlf.trainer(args,test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask, store_dir, 0)  
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # base
    parser.add_argument('--description_name', type=str, default='description')
    parser.add_argument('--description', type=str, default='hh')

    parser.add_argument('--alg', type=str, default="dl", help='dl or meta')
    parser.add_argument('--net', type=str,default="resnet18", help='choose model')
    parser.add_argument('--seed', type=int, default=1) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='0 = GPU.')
    parser.add_argument('--is_save_val_net', type=lambda x: (str(x).lower() == 'true'), default=True) # 是否保存验证集上最好的模型
    parser.add_argument('--save_path', type=str, default='tmp')
    parser.add_argument('--n_classes', type=int, default=1)# Total number of fc out_class
    parser.add_argument('--prefix', type=str, default='debug') # The network architecture   
    parser.add_argument('--load', '-f', type=str, default="", help='Load model from a .pth file')
    parser.add_argument('--load_interrupt_path', type=str, default="", help='Load model from a .pth file for interrupt recover')
    parser.add_argument('--best_acc', type=float, default=0.0)
    parser.add_argument('--best_epoch', type=int, default=0)
    parser.add_argument('--remark', type=str, default='base') # The network architecture

    # for dl/
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--resize', type=int, default=256)
    
    # meta 
    parser.add_argument('--n_way', type=int, help='n way', default=2)
    parser.add_argument('--k_shot', type=int, help='k shot for support set', default=5)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    parser.add_argument('--meta_size', type=int, help='meta batch size, namely meta_size',default=4)
    
    # 必须大于15，保证每个task,包含common所有类
    parser.add_argument('--test_k_shot', type=int, help='k shot for support set', default=15)
    parser.add_argument('--test_k_qry', type=int, help='k shot for query set', default=15)
    # 默认值为1，最好不要改动
    parser.add_argument('--test_meta_size', type=int, help='meta batch size, namely meta_size',default=1)
    
    parser.add_argument('--n_train_tasks', type=int, default=1000)# Total number of trainng tasks
    parser.add_argument('--n_val_tasks', type=int, default=250)# Total number of trainng tasks
    parser.add_argument('--n_test_tasks', type=int, default=4)# Total number of testing tasks
    parser.add_argument('--n_inner_updates', type=int, default=5)# Total number of image channles, 它增大，内存会随之增大
    parser.add_argument('--meta_lr', type=float, default=1e-3) # Learning rate for Adam weights
    parser.add_argument('--inner_lr', type=float, default=1e-1) # Learning rate for SGD weights
    
    args = parser.parse_args()
    
    utils.set_gpu(args.gpu)
    utils.set_seed(args.seed)
    
    project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication'
    begin_time = time.time()
    
    time_name=time.strftime('%Y-%m-%d-%H-%M-%S')
    store_dir =str(os.path.join(project_path, 'maml', args.save_path, args.net, args.prefix, time_name))
   
    utils.mkdir(store_dir)

    # 创建一个说明文件
    description_file = os.path.join(store_dir,  args.description_name)
    with open(str(description_file), 'w') as f:
        f.write(args.description)

    main()