import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
import os
from torch.utils.data import  DataLoader
import time
import logging
from copy import deepcopy
import csv

import common.originize_df as ori
from common.build_tasks import Train_task,Meta_Test_task
from common.dataloader import *
import dl.dl_func as dlf
import common.utils as utils


from common.meta.meta_comm import iMAML

from imaml.utils.utils import set_seed, set_gpu, check_dir, dict2tsv, BestTracker




def train(args, model, bayes_support_loader, bayes_query_loader, dataloader):

    loss_list = []
    acc_list = []
    grad_list = []
    with tqdm(dataloader, total=args.n_train_tasks // args.meta_size) as pbar:
        for batch_idx, batch in enumerate(pbar): # 为什么这句话会花超长时间
            random_index = random.randrange(0, len(bayes_support_loader), 1)

            bayes_choice_s = list(bayes_support_loader)[random_index]
            bayes_choice_q = list(bayes_query_loader)[random_index]
    
            loss_log, acc_log, grad_log = model.outer_loop_bayes(batch, bayes_choice_s, bayes_choice_q, is_train=True)

            loss_list.append(loss_log)
            acc_list.append(acc_log)
            grad_list.append(grad_log)
            pbar.set_description('loss = {:.4f} || acc={:.4f} || grad={:.4f}'.format(np.mean(loss_list), np.mean(acc_list), np.mean(grad_list)))
            if batch_idx >= args.n_train_tasks // args.meta_size:
                break

    loss = np.round(np.mean(loss_list), 4)
    acc = np.round(np.mean(acc_list), 4)
    grad = np.round(np.mean(grad_list), 4)

    return loss, acc, grad

@torch.no_grad()
def valid(args, model, dataloader):

    loss_list = []
    acc_list = []

    with tqdm(dataloader, total=args.num_valid_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):

            loss_log, acc_log = model.outer_loop(batch, is_train=False)

            loss_list.append(loss_log)
            acc_list.append(acc_log)
            pbar.set_description('loss = {:.4f} || acc={:.4f}'.format(np.mean(loss_list), np.mean(acc_list)))
            if batch_idx >= args.num_valid_batches:
                break

    loss = np.round(np.mean(loss_list), 4)
    acc = np.round(np.mean(acc_list), 4)

    return loss, acc


# @BestTracker
# def run_epoch(epoch, args, model, train_loader, valid_loader, test_loader):
def run_epoch(epoch, args, model, train_loader, bayes_support_loader, bayes_query_loader , test_loader_ls):
    res = OrderedDict()
    print('Epoch {}'.format(epoch))
    train_loss, train_acc, train_grad = train(args, model, bayes_support_loader, bayes_query_loader,train_loader)
    # valid_loss, valid_acc = valid(args, model, valid_loader)
    # test_loss, test_acc = valid(args, model, test_loader)

    res['epoch'] = epoch
    res['train_loss'] = train_loss
    res['train_acc'] = train_acc
    res['train_grad'] = train_grad
    # res['valid_loss'] = valid_loss
    # res['valid_acc'] = valid_acc
    # res['test_loss'] = test_loss
    # res['test_acc'] = test_acc

    # 保存每一个epoch的元参数
    meta_epoch_model = os.path.join(store_dir, f'meta_epoch_{epoch}.pth')
    torch.save(model.network.state_dict(), meta_epoch_model)

    meta_test_Q_values, meta_final_acc = dlf.trainer(args, test_loader_ls[0], test_loader_ls[1], test_loader_ls[2], store_dir, epoch)


    return res, meta_test_Q_values, meta_final_acc


def main(args):
    if args.alg=='MAML':
        pass
        # model = MAML(args)
    elif args.alg=='Reptile':
        # model = Reptile(args)
        pass
    elif args.alg=='Neumann':
        # model = Neumann(args)
        pass
    elif args.alg=='CAVIA':
        # model = CAVIA(args)
        pass
    elif args.alg=='iMAML':
        model = iMAML(args)
    elif args.alg=='FOMAML':
        # model = FOMAML(args)
        pass
    else:
        raise ValueError('Not implemented Meta-Learning Algorithm')

    if args.load:
        model.load()
    elif args.load_encoder:
        # 后续考虑怎么玩
        model.load_encoder()
    
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
    

    task_sppport_for_bayes = []
    task_query_for_bayes =[]
    df_for_bayes = pd.DataFrame()
    df_for_bayes = pd.concat([df_for_bayes,df_test_ls[0], df_test_ls[1]], ignore_index=True)
    for each_task in range(10):  # 目标任务
        task = Train_task(train_ways, train_shot, train_query,df_for_bayes, each_task, mode='train')
        task_sppport_for_bayes.append(task.support_roots)
        task_query_for_bayes.append(task.query_roots)


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
    train_support_loader = DataLoader(BasicDataset(train_support_fileroots_alltask, resize=args.resize), batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)
    train_query_loader = DataLoader(BasicDataset(train_query_fileroots_alltask, resize=args.resize), batch_size=args.meta_size, shuffle=True, num_workers=0, pin_memory=True)

    # bayes_task S和Q
    bayes_support_loader = DataLoader(BasicDataset(task_sppport_for_bayes,  resize=args.resize), batch_size=1, num_workers=0, pin_memory=True, shuffle=True)
    bayes_query_loader = DataLoader(BasicDataset(task_query_for_bayes,  resize=args.resize), batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
   
    best_pred_for_final = 0.0 #元测试中，测试任务 querry set上最好的值
    best_epoch_for_final = begin_epoch
    test_ls =[test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask] 
    for epoch in range(args.n_meta_epoch):
        # res, is_best = run_epoch(epoch, args, model, train_loader, valid_loader, test_loader)
        res, meta_test_Q_values, meta_final_acc= run_epoch(epoch, args, model, zip(train_support_loader, train_query_loader), bayes_support_loader, bayes_query_loader, test_ls)

        best_meta_test_value =  max(meta_test_Q_values)            
        is_best_for_final =best_meta_test_value > best_pred_for_final
        best_pred_for_final = max(best_pred_for_final, best_meta_test_value)
        if is_best_for_final :
            best_epoch_for_final = epoch

        with open(str(metric_dir_meta_test), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            
            best_val_acc = max(meta_test_Q_values)
            best_index = meta_test_Q_values.index(best_val_acc)
            
            data_row = [epoch] + meta_test_Q_values + [best_val_acc, best_index+1,best_meta_test_value,best_epoch_for_final, meta_final_acc]
            csv_write.writerow(data_row)

        # # 用于参考
        # with open(str(metric_dir_meta_test), 'a+') as f:
        #     csv_write = csv.writer(f, delimiter=',')
            
        #     best_test_acc = max(meta_final_values)
        #     best_test_index = meta_final_values.index(best_test_acc)
            
        #     data_row = [epoch] + meta_final_values + [best_test_acc,best_test_index+1,'final',best_epoch_for_final]
        #     csv_write.writerow(data_row)

        torch.cuda.empty_cache()
        # 默认不执行
        if args.lr_sched:
            model.lr_sched()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser('Gradient-Based Meta-Learning Algorithms')
    
    # base settings
    parser.add_argument('--description_name', type=str, default='description')
    parser.add_argument('--description', type=str, default='hh')
    
    # experimental settings
    # parser.add_argument('--data_set', type=str, default='Omniglot')
    # parser.add_argument('--data_path', type=str, default='../data/',
    #     help='Path of datasets.')
    # parser.add_argument('--result_path', type=str, default='./result')
    # parser.add_argument('--log_path', type=str, default='result.tsv')
    parser.add_argument('--load', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_encoder', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_path', type=str, default='')

    parser.add_argument('--num_workers', type=int, default=4,help='Number of workers for data loading (default: 4).')
    parser.add_argument('--resize', type=int, default=256)

    # for dl/
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--is_save_val_net', type=lambda x: (str(x).lower() == 'true'), default=False) # 是否保存验证集上最好的模型
    
    parser.add_argument('--seed', type=int, default=1) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='0 = GPU.')
    parser.add_argument('--save_path', type=str, default='tmp')
    parser.add_argument('--n_classes', type=int, default=1)# Total number of fc out_class
    parser.add_argument('--prefix', type=str, default='debug') # The network architecture   
    # parser.add_argument('--load', '-f', type=str, default="", help='Load model from a .pth file')
    parser.add_argument('--load_interrupt_path', type=str, default="", help='Load model from a .pth file for interrupt recover')
    parser.add_argument('--best_acc', type=float, default=0.0)
    parser.add_argument('--best_epoch', type=int, default=0)


    # training settings
    parser.add_argument('--n_meta_epoch', type=int, default=100,
        help='Number of epochs for meta train.') 
    # parser.add_argument('--batch_size', type=int, default=4,
    #     help='Number of tasks in a mini-batch of tasks (default: 4).')
    # parser.add_argument('--num_train_batches', type=int, default=100,
    #     help='Number of batches the model is trained over (default: 100).')
    # parser.add_argument('--num_valid_batches', type=int, default=150,
    #     help='Number of batches the model is validated over (default: 150).')

    # meta-learning settings
    # parser.add_argument('--num_shot', type=int, default=5,
    #     help='Number of support examples per class (k in "k-shot", default: 1).')
    # parser.add_argument('--num_way', type=int, default=5,
    #     help='Number of classes per task (N in "N-way", default: 5).')
    # parser.add_argument('--num_query', type=int, default=15,
    #     help='Number of query examples per class (k in "k-query", default: 15).')
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



    
    # imaml specific settings
    parser.add_argument('--lambda', type=float, default=2.0)# 并没有使用到啊擦
    parser.add_argument('--version', type=str, default='GD')
    parser.add_argument('--cg_steps', type=int, default=5) 
    
    # network settings
    parser.add_argument('--net', type=str, default='resnet18')
    # parser.add_argument('--n_conv', type=int, default=4)
    # parser.add_argument('--n_dense', type=int, default=0)
    # parser.add_argument('--hidden_dim', type=int, default=64)
    # parser.add_argument('--in_channels', type=int, default=1)
    # parser.add_argument('--hidden_channels', type=int, default=64,
    #     help='Number of channels for each convolutional layer (default: 64).')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication'
    begin_time = time.time()
    time_name=time.strftime('%Y-%m-%d-%H-%M-%S')
    store_dir =str(os.path.join(project_path, args.alg.lower(), args.save_path, args.net, time_name))
    store_dir_meta_test =str(os.path.join(project_path, args.alg.lower(), args.save_path, args.net, time_name, 'meta_test'))
    utils.mkdir(store_dir)
    utils.mkdir(store_dir_meta_test)

    # 创建一个说明文件
    description_file = os.path.join(store_dir,  args.description_name)
    with open(str(description_file), 'w') as f:
        f.write(args.description)
    # 创建一个记录meta_test的query set的acc
    metric_dir_meta_test = os.path.join(store_dir_meta_test, 'metric_meta_test_' +  time_name + '.csv')
    with open(str(metric_dir_meta_test), 'w') as f:
        fields = ['Meta_Epoch'] + list(range(1, args.n_epoch + 1, 1)) + ['best_val_acc', 'best_val_epoch', 'best_Q_acc', 'best_meta_epoch', 'final_test_acc']#记录内循环每次的acc
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)


    set_seed(args.seed)
    set_gpu(args.gpu)

    # check_dir(args)
    begin_epoch = 0
    # if args.load_interrupt_path != '':
    #     path = args.load_interrupt_path
    #     checkpoint = torch.load(path)
    #     net.load_state_dict(checkpoint['model'])
    #     meta_opt.load_state_dict(checkpoint['out_optimizer'])
    #     begin_epoch = checkpoint['epoch']
    #     print('中断恢复=', path)

    main(args)
