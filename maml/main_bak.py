import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/rp-project')

import os
import pandas as pd
import numpy as np
import argparse
import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import higher
import logging
import utils
from common.dataloader import *
from torch.utils.data import Dataset, DataLoader
from model.unet import UNet
from pathlib import Path
from tqdm import tqdm
from MTL.utils.losses import FocalLoss,CE_DiceLoss,LovaszSoftmax, CE_DiceLoss_OCT
from evl.evaluate import evaluate, single_batch_dsc
from sklearn.model_selection import KFold, GroupKFold

from common.build_tasks import Task

def unpack_batch(batch):
    # train_inputs, train_targets = batch[0]
    # train_inputs = train_inputs.cuda()
    # train_targets = train_targets.cuda()
    #
    # test_inputs, test_targets = batch[1]
    # test_inputs = test_inputs.cuda()
    #
    # test_targets = test_targets.cuda()

    device = torch.device('cuda')
    train_inputs, train_targets = batch[0]
    train_inputs = train_inputs.to(device=device, dtype=torch.float32)
    train_targets = train_targets.to(device=device, dtype=torch.long)

    test_inputs, test_targets = batch[1]
    test_inputs = test_inputs.to(device=device, dtype=torch.float32)
    test_targets = test_targets.to(device=device, dtype=torch.long)

    return train_inputs, train_targets, test_inputs, test_targets

def main(trainframein=None, testframein=None):
    
    time_name=time.strftime('%Y-%m-%d-%H-%M-%S') 
    project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/rp-project'

    if args.mode=='train':
        store_dir =str(Path(project_path, 'maml', args.save_path, args.prefix, time_name))
        metric_dir = Path(store_dir, 'metric_' +  time_name + '.txt')

        train_df_ori = pd.read_csv(project_path + '/DL/data_private/linux/train.csv')
        test_df_ori = pd.read_csv(project_path + '/DL/data_private/linux/test.csv')
        val_df_ori = pd.read_csv(project_path + '/DL/data_private/linux/val.csv')
        
        if args.use_data_percent != 1.0 :
            chosen_indices = range(len(train_df_ori['ID']))
            chosen_indices,_ = utils.data_split(chosen_indices, args.use_data_percent)
            train_df_ori = train_df_ori.iloc[chosen_indices]
            
            chosen_indices = range(len(test_df_ori['ID']))
            chosen_indices,_ = utils.data_split(chosen_indices, args.use_data_percent)
            test_df_ori = test_df_ori.iloc[chosen_indices]
            
            chosen_indices = range(len(val_df_ori['ID']))
            chosen_indices,_ = utils.data_split(chosen_indices, args.use_data_percent)
            val_df_ori = val_df_ori.iloc[chosen_indices]
        
        tldframe = pd.read_csv(args.tl_data_csv)
        if args.isTL and args.real_patch_path:
            df_real_patch = pd.read_csv(args.real_patch_path)
            tldframe = pd.concat([tldframe, df_real_patch], ignore_index=True)
        
        idx = list(train_df_ori['ID'])
        train_df = tldframe.loc[tldframe['ID'] == idx[0]]
        for i in range(len(idx) - 1):   
            train_df = pd.concat([train_df, tldframe.loc[tldframe['ID'] == idx[i + 1]]] )
        
        idx = list(val_df_ori['ID']) 
        val_df = tldframe.loc[tldframe['ID'] == idx[0]] 
        for i in range(len(idx) - 1):   
            val_df = pd.concat([val_df, tldframe.loc[tldframe['ID'] == idx[i + 1]]] )
        
        idx = list(test_df_ori['ID']) 
        test_df = tldframe.loc[tldframe['ID'] == idx[0]] 
        for i in range(len(idx) - 1):   
            test_df = pd.concat([test_df, tldframe.loc[tldframe['ID'] == idx[i + 1]]] )    
        
        trainframe = train_df
        testframe = val_df
        
    elif args.mode == 'k_fold':
        store_dir =str(Path(project_path, 'maml', args.save_path, args.prefix, str(args.index_fold) + '_fold'))
        metric_dir = Path(store_dir, 'metric_' +  time_name + '.txt')
        trainframe = trainframein
        testframe = testframein    
        
        
    utils.mkdir(store_dir)
    # writer = SummaryWriter(store_dir)
    


    train_classes = np.unique(trainframe["ID"])
    train_classes = list(train_classes)
    all_test_classes = np.unique(testframe["ID"])
    all_test_classes = list(all_test_classes)
    
    train_ways = args.n_way  # ，每个training task包含的类别数量
    train_shot = args.k_shot  # ，每个training task里每个类别的training sample的数量
    train_query = args.k_shot
    
    test_shot = args.k_qry  # ，每个测试任务，每个类别包含的训练样本的数量
    test_ways = args.n_way  # 
    test_query = args.k_shot  # ， 每个测试任务，每个类别包含的测试样本的数量
    
    train_support_fileroots_alltask, train_query_fileroots_alltask = [], []

    for each_task in range(args.n_train_tasks):  # num_train_task 训练任务的总数
        task = Task(train_classes, train_ways, train_shot, train_query,trainframe)
        train_support_fileroots_alltask.append(task.support_roots)
        train_query_fileroots_alltask.append(task.query_roots)
    
    val_support_fileroots_alltask, val_query_fileroots_alltask = [], []
    for each_task in range(args.n_val_tasks):
        task = Task(val_classes, train_ways, train_shot, train_query,trainframe)
        val_support_fileroots_alltask.append(task.support_roots)
        val_query_fileroots_alltask.append(task.query_roots)


    test_query_fileroots_alltask, test_support_fileroots_all_task = [], []

    for each_task in range(args.n_test_tasks):  # 测试任务总数
        test_task = Task(all_test_classes, test_ways, test_shot, test_query, testframe)
        test_query_fileroots_alltask.append(test_task.query_roots)
        test_support_fileroots_all_task.append(test_task.support_roots)
    
    # train_task S 和Q
    train_support_loader = DataLoader(BasicDataset(train_support_fileroots_alltask), batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)
    train_query_loader = DataLoader(BasicDataset(train_query_fileroots_alltask), batch_size=args.meta_size, shuffle=True, num_workers=0, pin_memory=True)

    # val_task S 和Q
    val_support_loader = DataLoader(BasicDataset(train_support_fileroots_alltask), batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)
    val_query_loader = DataLoader(BasicDataset(train_query_fileroots_alltask), batch_size=args.meta_size, shuffle=True, num_workers=0, pin_memory=True)    
    
    
    test_support_loader = DataLoader(BasicDataset(test_support_fileroots_all_task), batch_size=args.test_meta_size, shuffle=True, num_workers=0, pin_memory=True)
    test_query_loader = DataLoader(BasicDataset(test_query_fileroots_alltask), batch_size=args.test_meta_size, shuffle=True, num_workers=0, pin_memory=True)
    
    
        
    net = UNet(n_channels=args.n_channels, n_classes=args.n_classes, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    
    
    meta_opt = optim.SGD(net.parameters(), lr=args.meta_lr)
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
        
    net.to(device=device)
    
    best_dice = args.best_dice
    best_std = 0.0
    for epoch in range(begin_epoch, 200):
        ave_dice_train, train_dice_last = train(zip(train_support_loader, train_query_loader), net, device, meta_opt, epoch)
        ave_dice_test, std = test(zip(test_support_loader, test_query_loader), net, device, epoch)  
                    
        is_best = ave_dice_test > best_dice
        best_dice = max(ave_dice_test, best_dice)
        if is_best:
            best_std = std
            best_epoch = epoch
            torch.save(net.state_dict(),Path(store_dir, 'best_model.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved!')
    
        with open(str(metric_dir), 'a+') as f:
            f.write('EPOCH:' + str(epoch) + ', ')
            f.write('train_dice_epoch:' + str(round(float(ave_dice_train) , 4)) + ', ')
            f.write('train_dice_last:' + str(round(float(train_dice_last) , 4)) + ', ')
            f.write('test_dice:' + str(round(float(ave_dice_test) , 4)) + '±' + str(round(float(std), 6)) +', ')
            f.write('best_epoch:' + str(best_epoch) +', ')
            f.write('best_test_dice:' + str(round(float(best_dice) , 4)) + '±' + str(round(float(best_std), 6)))
            f.write('run_time_hours:' + str(round((time.time() - begin_time) / 3600 , 2)))
            f.write('\n')
        
        # 每次epoch保存一个以便异常回复
        # torch.save(net.state_dict(),Path(store_dir, 'last_epoch.pth'))
        state = {'model':net.state_dict(), 'out_optimizer':meta_opt.state_dict(),'epoch':epoch}     
        path = os.path.join(store_dir, 'last_epoch_state.pth')
        torch.save(state, path)
        

def train(db, net, device, meta_opt, epoch):
    net.train()
    
    CD = CE_DiceLoss_OCT(args.n_classes)

    qry_dsc_all_tasks = []
    tqdm_train = tqdm(db)
    for batch_idx, batch in enumerate(tqdm_train, 1):
        support_x, support_y, query_x, query_y = unpack_batch(batch)
        task_num, setsz, c_, h, w = support_x.size()
        querysz = query_x.size(1)
        
        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        # 每一个epoch都实例化一个，不用从上一个epoch开始
        inner_opt = torch.optim.SGD(net.parameters(), lr=args.inner_lr)
        qry_losses = []
        qry_dscs = []
        
        meta_opt.zero_grad()
        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(args.n_inner_updates):
                    spt_logits = fnet(support_x[i])
                    spt_loss = CD(spt_logits, support_y[i])

                    diffopt.step(spt_loss)
                    
                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                qry_logits = fnet(query_x[i])
                qry_loss = CD(qry_logits, query_y[i])
                qry_losses.append(qry_loss.detach())
                qry_dsc = single_batch_dsc(qry_logits, query_y[i], args.n_classes)

                qry_dscs.append(qry_dsc)
                
                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                qry_loss.backward() # 为什么要在这呢
                
        meta_opt.step()
        qry_dscs_last = qry_dscs[-1]
        qry_losses = sum(qry_losses) / task_num
        # qry_dscs = 100. * sum(qry_dscs) / task_num # .* 和*有什么区别吗？没有吧
        qry_dscs = sum(qry_dscs) / task_num
        
        qry_dsc_all_tasks.append(qry_dscs)
        
        tqdm_train.set_description('Training_Tasks Epoch {}, batch_idx {}, DSC={:.4f}, , Loss={:.4f}'.format(epoch, batch_idx, qry_dscs.item(), qry_losses))
        if batch_idx % 5 == 0:
            print('\n')
            logging.info(f'step {batch_idx + 1} training: {qry_dscs}!')
            
    ave_qry_dscs_all_tasks = np.array(qry_dsc_all_tasks).mean()
    return ave_qry_dscs_all_tasks, qry_dsc_all_tasks[-1]

        
def test(db, net, device, epoch):
    net = deepcopy(net)
    net.train()
    CD = CE_DiceLoss_OCT(args.n_classes)

    qry_losses = []
    qry_dscs = []
    
    tqdm_test = tqdm(db)
    for batch_idx, batch in enumerate(tqdm_test, 1):
        support_x, support_y, query_x, query_y = unpack_batch(batch)
        task_num, setsz, c_, h, w = support_x.size()
        querysz = query_x.size(1)
        
        n_inner_iter = args.n_inner_updates
        inner_opt = torch.optim.SGD(net.parameters(), lr=args.inner_lr)
        
        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(support_x[i])
                    spt_loss = CD(spt_logits, support_y[i])
                    diffopt.step(spt_loss)
                    
                
                # The query loss and acc induced by these parameters.
                qry_logits = fnet(query_x[i]).detach()
                qry_loss = CD(qry_logits, query_y[i])
                qry_losses.append(qry_loss.detach()) 
                qry_dsc = single_batch_dsc(qry_logits, query_y[i], args.n_classes)
                qry_dscs.append(qry_dsc.detach())
            
        tqdm_test.set_description('Testing_Tasks Epoch {}, DSC={:.4f}'.format(epoch, qry_dsc.item()))
            
    # qry_losses =  torch.cat(qry_losses).mean().item()       
    # qry_dscs = 100. * torch.cat(qry_dscs).float().mean().item()
    ave_dice = np.array(qry_dscs).mean()
    std = np.array(qry_dscs).std()    
    
    return ave_dice, std
 
 
def begin_k_fold():
    tldframe = pd.read_csv(args.tl_data_csv)

    if args.isTL and args.real_patch_path:
        df_real_patch = pd.read_csv(args.real_patch_path)
        tldframe = pd.concat([tldframe, df_real_patch], ignore_index=True)
        
    k_fold_num = args.k_fold_num
    # kf = KFold(n_splits=k_fold_num, random_state=None) # 4折
    kf = GroupKFold(n_splits=k_fold_num)
    dframe = pd.read_csv(args.all_data_csv_path)
    k_fold_count = 0
    # 是否随机有待验证，不能随机
    # dframe = dframe.sample(int(dframe.shape[0] * args.use_data_percent))
    train_df_list =[]
    test_df_list = []
    groups = list(dframe['Eye'])
    # for train_index, test_index in kf.split(dframe):
    for train_index, test_index in kf.split(dframe,groups=groups):
        k_fold_count += 1
        print('\n{} of kfold {}'.format(k_fold_count,kf.n_splits))
        train_df_ori = dframe.iloc[train_index,:]
        test_df_ori = dframe.iloc[test_index,:]
        
        # 牛逼不解释
        idx = list(train_df_ori['ID'])  
        train_df = tldframe.loc[tldframe['ID'] == idx[0]]
        for i in range(len(idx) - 1):   
            train_df = pd.concat([train_df, tldframe.loc[tldframe['ID'] == idx[i + 1]]] )

        idx = list(test_df_ori['ID']) 
        test_df = tldframe.loc[tldframe['ID'] == idx[0]] 
        for i in range(len(idx) - 1):   
            test_df = pd.concat([test_df, tldframe.loc[tldframe['ID'] == idx[i + 1]]] )

        train_df_list.append(train_df)
        test_df_list.append(test_df)
        
    # return train_df_list, test_df_list
    main(train_df_list[args.index_fold-1], test_df_list[args.index_fold-1] )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # base
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'k_fold'])
    parser.add_argument('--seed', type=int, default=1) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='0 = GPU.')
    parser.add_argument('--save_path', type=str, default='tmp') # The network architecture
    parser.add_argument('--prefix', type=str, default='debug') # The network architecture    
    parser.add_argument('--n_classes', type=int, default=1)# Total number of u-net out_class
    parser.add_argument('--n_channels', type=int, default=3)# Total number of image channles
    parser.add_argument('--load', '-f', type=str, default="", help='Load model from a .pth file')
    parser.add_argument('--load_interrupt_path', type=str, default="", help='Load model from a .pth file for interrupt recover')
    parser.add_argument('--best_dice', type=float, default=0.0)
    parser.add_argument('--best_epoch', type=int, default=0)

    # meta 
    parser.add_argument('--n_way', type=int, help='n way', default=2)
    parser.add_argument('--k_shot', type=int, help='k shot for support set', default=4)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=4)
    parser.add_argument('--meta_size', type=int, help='meta batch size, namely meta_size',default=4)
    parser.add_argument('--n_train_tasks', type=int, default=1000)# Total number of trainng tasks
    parser.add_argument('--n_test_tasks', type=int, default=200)# Total number of testing tasks
    parser.add_argument('--n_inner_updates', type=int, default=4)# Total number of image channles, 它增大，内存会随之增大
    parser.add_argument('--meta_lr', type=float, default=1e-3) # Learning rate for SS weights
    parser.add_argument('--inner_lr', type=float, default=1e-1) # Learning rate for SS weights
    
    # k-fold
    parser.add_argument('--k_fold_num', '-kf', type=int, default=4, help='k fold')
    parser.add_argument('--all_data_csv_path', '-as', type=str, default='')
    parser.add_argument('--use_data_percent', '-udp', metavar='UDP', type=float, default=1, help='use data percent')
    parser.add_argument('--index_fold', type=int, default=1, help='the index fold.')
    parser.add_argument('--tl_data_csv', type=str, default='')
    parser.add_argument('--real_patch_path', type=str, default='')
    parser.add_argument('--isTL', type=int, default=1, help='whether use real_patch')
    
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    utils.set_gpu(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
     
    # Set manual seed for PyTorch
    utils.set_seed(args.seed)
    
    begin_time = time.time()
    
    if args.mode == 'train':
        main()
    elif args.mode == 'k_fold':
        begin_k_fold()
