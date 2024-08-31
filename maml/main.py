import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')


import os
import argparse
import time
import csv
import logging
import shutil

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


def main_with_val_task():

    metric_dir = os.path.join(store_dir, 'metric_' +  time_name + '.csv')

    
    # 创建
    with open(str(metric_dir), 'w') as f:
        fields = ['Meta_Epoch','train_acc_last_task', 'val_acc_ave', 'best_val_acc','best_epoch']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)
        
    # 创建一个记录每一次梯度的，用于观察
    metric_dir_loop = os.path.join(store_dir, 'metric_' +  time_name + '_loop.csv')
    with open(str(metric_dir_loop), 'w') as f:
        fields = ['Meta_Epoch'] + list(range(1, args.n_inner_updates+1, 1)) + ['best_acc', 'best_epoch']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)
          
    # 创建一个记录meta_test的query set的acc
    metric_dir_meta_test = os.path.join(store_dir_meta_test, 'metric_meta_test_' +  time_name + '.csv')
    with open(str(metric_dir_meta_test), 'w') as f:
        fields = ['Meta_Epoch'] + list(range(1, 21, 1)) + ['best_val_acc', 'best_val_epoch', 'best_Q_acc', 'best_meta_epoch']#记录内循环每次的acc
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)
          

    df_train, df_val, df_test_ls,train_classes, val_classes, test_classes = ori.originze_df(project_path)

    
    train_ways = args.n_way  # ，每个training task包含的类别数量
    train_shot = args.k_shot  # ，每个training task里每个类别的training sample的数量
    train_query = args.k_shot
    
    train_support_fileroots_alltask, train_query_fileroots_alltask = [], []
    
    for each_task in range(args.n_train_tasks):  # num_train_task 训练任务的总数
        task = Train_task(train_classes, train_ways, train_shot, train_query,df_train,each_task, mode='train')
        train_support_fileroots_alltask.append(task.support_roots)
        train_query_fileroots_alltask.append(task.query_roots)
    
    val_support_fileroots_alltask, val_query_fileroots_alltask = [], []
    for each_task in range(args.n_val_tasks):  
        task = Train_task(val_classes, train_ways, train_shot, train_query,df_val, each_task, mode='val')
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
    
    begin_epoch = 0
    best_pred = args.best_acc
    best_epoch= begin_epoch
    meta_ob = meta_comm(args)

    best_pred_for_final_val = 0.0 #元测试中，测试任务 querry set上最好的值
    best_epoch_for_final_valset = begin_epoch
    
    # 先保存一个，用于判断是否更新
    meta_potential_for_val_task =  os.path.join(store_dir, 'meta_potential_model_for_meta_val_' + time_name + '.pth')    
    torch.save(meta_ob.model.state_dict(), meta_potential_for_val_task)
    time_ct = int(os.stat(meta_potential_for_val_task).st_ctime)

    for epoch in range(begin_epoch, 200):
        ave_acc_all_tasks, acc_last_task = meta_ob.train(zip(train_support_loader, train_query_loader), epoch)
        
        # 保存每一个epoch的元参数
        meta_epoch_model = os.path.join(store_dir, 'meta_epoch_model_' + str(epoch) + '.pth')
        torch.save(meta_ob.model.state_dict(), meta_epoch_model)

        val_value = meta_ob.val(zip(val_support_loader, val_query_loader), epoch)
        
        is_best = val_value[0] > best_pred
        best_pred = max(best_pred, val_value[0])
        
        if is_best:
            best_epoch = epoch
            
            #保存元训练过程中，验证任务上最具潜力的模型
            # meta_potential_for_val_task =  os.path.join(store_dir, 'meta_potential_model_for_meta_val_' + time_name + '.pth')            
            torch.save(meta_ob.model.state_dict(), meta_potential_for_val_task)
            logging.info(f'Checkpoint {epoch + 1} saved!')
                    
        with open(str(metric_dir), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            data_row = [epoch, acc_last_task, str(val_value[0]) + '±' + str (val_value[1]), best_pred, best_epoch]
            csv_write.writerow(data_row)
            
        with open(str(metric_dir_loop), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            data_row = [epoch] + val_value[2]
            data_row.append(best_pred)
            data_row.append(best_epoch)
            csv_write.writerow(data_row)
        
        # 元测试       
        test_model = deepcopy(net)
        params = deepcopy(torch.load(meta_epoch_model))
        # test_model.load_state_dict(params)

        # 返回list, 1.验证集（querry set)的每个epoch的ACC
        meta_test_values, meta_final_values = dlf.dl_for_meta_test(args,test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask, store_dir_meta_test, device, test_model, params, epoch)
        

        best_meta_test_Q_value =  max(meta_test_values)            
        is_best_for_final_valset =best_meta_test_Q_value > best_pred_for_final_val
        best_pred_for_final_val = max(best_pred_for_final_val, best_meta_test_Q_value)
        if is_best_for_final_valset :
            best_epoch_for_final_valset = epoch

        with open(str(metric_dir_meta_test), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            
            best_val_acc = max(meta_test_values)
            best_index = meta_test_values.index(best_val_acc)
            
            data_row = [epoch] + meta_test_values + [best_val_acc, best_index+1,best_meta_test_Q_value,best_epoch_for_final_valset]
            csv_write.writerow(data_row)

        # 用于参考
        with open(str(metric_dir_meta_test), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            
            best_test_acc = max(meta_final_values)
            best_test_index = meta_final_values.index(best_test_acc)
            
            data_row = [epoch] + meta_final_values + [best_test_acc,best_test_index+1,'final',best_epoch_for_final_valset]
            csv_write.writerow(data_row)

        del test_model
        


def main_without_val_task():


    # 创建
    metric_dir = os.path.join(store_dir, 'metric_' +  time_name + '.csv')
    with open(str(metric_dir), 'w') as f:
        fields = ['Meta_Epoch','train_acc_last_task', 'val_acc_ave', 'best_val_acc','best_epoch']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)
        
    # 创建一个记录每一次梯度的，用于观察
    metric_dir_loop = os.path.join(store_dir, 'metric_' +  time_name + '_loop.csv')
    with open(str(metric_dir_loop), 'w') as f:
        fields = ['Meta_Epoch'] + list(range(1, args.n_inner_updates+1, 1)) + ['best_acc', 'best_epoch']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)
          
    # 创建一个记录meta_test的query set的acc
    metric_dir_meta_test = os.path.join(store_dir_meta_test, 'metric_meta_test_' +  time_name + '.csv')
    with open(str(metric_dir_meta_test), 'w') as f:
        fields = ['Meta_Epoch'] + list(range(1, args.n_epoch + 1, 1)) + ['best_val_acc', 'best_val_epoch', 'best_Q_acc', 'best_meta_epoch', 'final_test_acc']#记录内循环每次的acc
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)
          
    target_classes = ['normal', 'PIC', 'PVRL',  'RP'] 
    df_train, df_test_ls = ori.originze_df_maml_four(project_path, target_classes)

    train_ways = args.n_way  # ，每个training task包含的类别数量
    train_shot = args.k_shot  # ，每个training task里每个类别的training sample的数量
    train_query = args.k_shot
    
    train_support_fileroots_alltask, train_query_fileroots_alltask = [], []
    
    for each_task in range(args.n_train_tasks):  # num_train_task 训练任务的总数
        task = Train_task(train_ways, train_shot, train_query,df_train,each_task, mode='train')
        train_support_fileroots_alltask.append(task.support_roots)
        train_query_fileroots_alltask.append(task.query_roots)
    

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
    
    begin_epoch = 0
    best_pred = args.best_acc
    best_epoch= begin_epoch
    meta_ob = meta_comm(args)

    best_pred_for_final_val = 0.0 #元测试中，测试任务 querry set上最好的值
    best_epoch_for_final_valset = begin_epoch
    
    # 先保存一个，用于判断是否更新
    meta_potential_for_val_task =  os.path.join(store_dir, 'meta_potential_model_for_meta_val_' + time_name + '.pth')    
    torch.save(meta_ob.model.state_dict(), meta_potential_for_val_task)
    time_ct = int(os.stat(meta_potential_for_val_task).st_ctime)

    for epoch in range(begin_epoch, 200):
        
        ave_acc_all_tasks, acc_last_task = meta_ob.train(zip(train_support_loader, train_query_loader), epoch)

        # 保存每一个epoch的元参数
        meta_epoch_model = os.path.join(store_dir, f'meta_epoch_{epoch}.pth')
        torch.save(meta_ob.model.state_dict(), meta_epoch_model)

        # 返回list, 1.验证集（querry set)的每个epoch的ACC
        meta_test_Q_values, final_test_acc = dlf.trainer(args,test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask, store_dir, epoch)  

        best_meta_test_Q_value =  max(meta_test_Q_values)            
        is_best_for_final_valset =best_meta_test_Q_value > best_pred_for_final_val
        best_pred_for_final_val = max(best_pred_for_final_val, best_meta_test_Q_value)
        if is_best_for_final_valset :
            best_epoch_for_final_valset = epoch

        with open(str(metric_dir_meta_test), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            
            best_val_acc = max(meta_test_Q_values)
            best_index = meta_test_Q_values.index(best_val_acc)
            
            data_row = [epoch] + meta_test_Q_values + [best_val_acc, best_index+1,best_meta_test_Q_value,best_epoch_for_final_valset, final_test_acc]
            csv_write.writerow(data_row)

        # 默认不执行
        if args.lr_sched:
            meta_ob.lr_sched()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # base
    parser.add_argument('--description_name', type=str, default='description')
    parser.add_argument('--description', type=str, default='hh')
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
    parser.add_argument('--is_save_val_net', type=lambda x: (str(x).lower() == 'true'), default=False) # 是否保存验证集上最好的模型
    parser.add_argument('--alg', type=str, default="maml", help='dl, maml, imaml')
    parser.add_argument('--lr_sched', type=lambda x: (str(x).lower() == 'true'), default=False)
    # for dl/
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--resize', type=int, default=256)
    
    # meta 
    parser.add_argument('--n_way', type=int, help='n way', default=2)
    parser.add_argument('--k_shot', type=int, help='k shot for support set', default=15)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
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
    store_dir =str(os.path.join(project_path, 'maml', args.save_path, args.net, time_name))

    store_dir_meta_test =str(os.path.join(project_path, 'maml', args.save_path, args.net, time_name, 'meta_test'))
    utils.mkdir(store_dir)
    utils.mkdir(store_dir_meta_test)

    # 创建一个说明文件
    description_file = os.path.join(store_dir,  args.description_name)
    with open(str(description_file), 'w') as f:
        f.write(args.description)


    main_without_val_task()