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
from common.dl_comm import dl_comm
from common.meta.meta_comm import meta_comm

#TEST begin++++++++++++++++++++++++
import common.oct_pro as oct_pro

from dl.dataloader_OCTimage import oct_img
#Test end--------------------------


def main():
 
    df_train, df_val, df_test_ls,train_classes, val_classes, test_classes = ori.originze_df(project_path)

    
    train_ways = args.n_way  # ，每个training task包含的类别数量
    train_shot = args.k_shot  # ，每个training task里每个类别的training sample的数量
    train_query = args.k_shot
    
    train_support_fileroots_alltask, train_query_fileroots_alltask = [], []
    
    for each_task in range(args.n_train_tasks):  # num_train_task 训练任务的总数
        task = Train_task(train_classes, train_ways, train_shot, train_query,df_train)
        train_support_fileroots_alltask.append(task.support_roots)
        train_query_fileroots_alltask.append(task.query_roots)
    
    val_support_fileroots_alltask, val_query_fileroots_alltask = [], []
    for each_task in range(args.n_val_tasks):  
        task = Train_task(val_classes, train_ways, train_shot, train_query,df_val)
        val_support_fileroots_alltask.append(task.support_roots)
        val_query_fileroots_alltask.append(task.query_roots) 


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
    
    # train_task S 和Q   
    train_support_loader = DataLoader(BasicDataset(train_support_fileroots_alltask), batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)
    train_query_loader = DataLoader(BasicDataset(train_query_fileroots_alltask), batch_size=args.meta_size, shuffle=True, num_workers=0, pin_memory=True)
    
    # val_task S 和Q
    val_support_loader = DataLoader(BasicDataset(val_support_fileroots_alltask), batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)
    val_query_loader = DataLoader(BasicDataset(val_query_fileroots_alltask), batch_size=args.meta_size, shuffle=False, num_workers=0, pin_memory=True)    
    
    # test_task S 和Q
    # test_support_loader = DataLoader(BasicDataset(test_support_fileroots_all_task), batch_size=args.test_meta_size, shuffle=True, num_workers=0, pin_memory=True)
    # test_query_loader = DataLoader(BasicDataset(test_query_fileroots_alltask), batch_size=args.test_meta_size, shuffle=False, num_workers=0, pin_memory=True)
                

    dl_ob = dl_comm(net, device)
    
    # df_train_data, df_val_data, df_test_data=oct_pro.data_choose()
    # test_set = oct_img(dataframe=df_test_data, mode='test')
    # test_loader = DataLoader(test_set, shuffle=False, drop_last=True, batch_size = 1, num_workers=4, pin_memory=True)
    

    # test_data_list = final_test_alltask[0]
    test_data_list = test_query_fileroots_alltask[0]
    test_set = DLdataset(test_data_list, mode='test')
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, batch_size = 4, num_workers=4, pin_memory=True)
    
    test_values = dl_ob.test(test_loader)
    
        
               

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # base
    parser.add_argument('--net', type=str,default="resnet18", help='choose model')
    parser.add_argument('--seed', type=int, default=1) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='0 = GPU.')
    parser.add_argument('--save_path', type=str, default='tmp') # The network architecture
    parser.add_argument('--n_classes', type=int, default=1)# Total number of fc out_class
    parser.add_argument('--prefix', type=str, default='debug') # The network architecture   
    parser.add_argument('--load', '-f', type=str, default="", help='Load model from a .pth file')
    parser.add_argument('--load_interrupt_path', type=str, default="", help='Load model from a .pth file for interrupt recover')
    parser.add_argument('--best_acc', type=float, default=0.0)
    parser.add_argument('--best_epoch', type=int, default=0)
    
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    utils.set_seed(args.seed)
    
    net = torchvision.models.resnet18(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 2)


    net.to(device)
    # 分割都是用SGD 嘛？？？？？
    meta_opt = optim.Adam(net.parameters(), lr=1e-3)

    
    begin_epoch = 0
    if args.load_interrupt_path != '':
        path = args.load_interrupt_path
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['model'])
        meta_opt.load_state_dict(checkpoint['out_optimizer'])
        begin_epoch = checkpoint['epoch']
        print('中断恢复=', path)
    elif args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}') 
    
    
    
    project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication'
    begin_time = time.time()
    
    time_name=time.strftime('%Y-%m-%d-%H-%M-%S')
    store_dir =str(os.path.join(project_path, 'maml', args.save_path, args.net, time_name))

    main()