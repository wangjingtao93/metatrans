import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')


import os
import pandas as pd
import numpy as np
import argparse
import time
import random
from tqdm import tqdm
import csv

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import higher
import logging
from torch.utils.data import Dataset, DataLoader
import torchvision.models

from common.dataloader import *

import common.utils as utils
from common.build_tasks import Train_task,Meta_Test_task
import common.eval as evl
import common.originize_df as ori
from common.dl_comm import dl_comm
from copy import deepcopy

from sklearn.metrics import confusion_matrix, auc, roc_curve, recall_score, precision_score, f1_score, cohen_kappa_score,accuracy_score

# # 小函数
# # 更改病人id
# def change_hei_id(df_ori, df_tar):
#     patient_len = len(np.unique(list(df_ori['Patient'])))
    
#     tar_patient_index = list(df_tar['Patient'])
#     idx_new = []
#     for i in tar_patient_index:
#         idx_new.append(i + patient_len)
    
#     df_tar['Patient'] = idx_new
#     return df_tar


    
# # 小函数
# # 生成label_key, key = class_name, value=id
# def classname_with_classID(pdframe):
#     all_class_name = np.unique(pdframe["Class"])
    
#     dict_labels = {}
#     ls_labels = []
#     for label, class_name in enumerate(all_class_name):
#         dict_labels[class_name] = label
#         ls_labels.append(label)

#     return dict_labels, ls_labels

# # 小函数，增加label列
# def add_classID(pdframe, dict_labels):

#     label_list = []
#     for class_name in pdframe['Class']:
#         label_list.append(dict_labels[class_name])
    
#     pdframe['Label'] = label_list
#     return pdframe

# # 随机划分类别
# def data_split(full_list, ratio, shuffle=False):
#     """
#     数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
#     :param full_list: 数据列表
#     :param ratio:     子列表1
#     :param shuffle:   子列表2
#     :return:
#     """
#     n_total = len(full_list)
#     offset = int(n_total * ratio)
#     if n_total == 0 or offset < 1:
#         return [], full_list
#     if shuffle:
#         random.shuffle(full_list)
#     sublist_1 = full_list[:offset]
#     sublist_2 = full_list[offset:]
#     return sublist_1, sublist_2




# def originze_df():
    
#     df_triton_train = pd.read_csv(os.path.join(project_path, 'data/linux/triton/6_1_1/train.csv'))
#     df_triton_train_pvrl = df_triton_train[df_triton_train['Class'] == 'PVRL']
          
#     df_triton_val = pd.read_csv(os.path.join(project_path,'data/linux/triton/6_1_1/val.csv'))
#     df_triton_val_pvrl = df_triton_val[df_triton_val['Class'] == 'PVRL']
    
#     df_triton_test = pd.read_csv(os.path.join(project_path,'data/linux/triton/6_1_1/test.csv'))
#     df_triton_test_pvrl = df_triton_test[df_triton_test['Class'] == 'PVRL']
    
    
#     # add heidelberg
#     df_hei = pd.read_csv(os.path.join(project_path,'data/linux/heidelberg/data.csv'))   
    
#     df_hei_pvrl = df_hei[df_hei['Class'] == 'PVRL']
    
#     df_hei_val_pvrl = df_hei_pvrl[df_hei_pvrl['Patient'] == 6]
#     df_hei_test_pvrl = df_hei_pvrl[df_hei_pvrl['Patient'] == 7]
    
#     df_hei_train_pvrl = df_hei_pvrl.drop(df_hei_val_pvrl.index).drop(df_hei_test_pvrl.index)
    
#     df_hei_train_pvrl = change_hei_id(df_triton_train_pvrl, df_hei_train_pvrl)
#     df_hei_val_pvrl = change_hei_id(df_triton_val_pvrl, df_hei_val_pvrl)
#     df_hei_test_pvrl = change_hei_id(df_triton_test_pvrl, df_hei_test_pvrl)
    
#     df_train = pd.concat([df_triton_train, df_hei_train_pvrl],ignore_index=True)  
#     df_val = pd.concat([df_triton_val, df_hei_val_pvrl],ignore_index=True) 
#     df_test = pd.concat([df_triton_test, df_hei_test_pvrl],ignore_index=True)
    

#     classid_dict, classid_ls = classname_with_classID(df_train)

#     # 增加label列
#     df_train = add_classID(df_train, classid_dict)
#     df_val = add_classID(df_train, classid_dict)
#     df_test = add_classID(df_test, classid_dict)
    
    
#     # 方案一：
#     # train_classes:common
    
#     # 方案二
#     # train_classes: (1-12), val_classes:common(13~15), test_class:rare+others(common 1~15)
    
#     # rare未出现在元训练，将train_data和val_data里的rare加到test_data,可以增加病人数量，更牛逼
#     # df_test = pd.concat([df_test, df_train['PVRL'], df_val['PVRL']],ignore_index=True)
#     df_test_ls = [df_train, df_val, df_test]
    
#     # 先drop,后面不drop,提出创新点，增加模块或机制，提高含PVRL类的任务的权重--》
#     # 提高某一类别的分类准去率
#     df_train = df_train.drop(df_train[df_train['Class'] == 'PVRL'].index)
#     df_val = df_val.drop(df_val[df_val['Class'] == 'PVRL'].index)
    
    
#     # 划分train_classes和val_classes
#     class_names = np.unique(list(df_train['Class']))
#     train_classes, val_classes = data_split(list(class_names), 0.8,shuffle=False)

#     train_class_id = []
#     val_class_id = []
#     # for i in val_classes:
#     #     df_train.drop(i)
#     #     val_class_id.append(classid_dict[i])

    
#     # for i in train_classes:
#     #     df_val.drop(i)
#     #     train_class_id.append(classid_dict[i])
        
    
#     for id, class_name in enumerate(classid_dict):
#         if class_name in val_classes:
#             val_class_id.append(id)
            
#             # 可以充分利用数据进行元训练
#             # 但是考虑到，元训练的数据，基本都是common类，似乎用太多common类对结果并不会有太大提升
#             df_val = pd.concat([df_val, df_train[df_train['Class'] == class_name]],ignore_index=True)
            
#             df_train.drop(df_train[df_train['Class'] == class_name].index)
        
#         if class_name in  train_classes:
#             train_class_id.append(id)
            
#             # 可以充分利用数据进行元训练
#             # 但是考虑到，元训练的数据，基本都是common类，似乎用太多common类对结果并不
#             df_train = pd.concat([df_train, df_val[df_val['Class'] == class_name]],ignore_index=True)
            
#             df_val.drop(df_val[df_val['Class'] == class_name].index)





#     # 方案三:
#     # train_classes:rare + others(common:1-12), val_classes:rare+others(common:13~15), test_class:rare+others(common 1~15)   

#     return df_train, df_val, df_test_ls, train_class_id, val_class_id, classid_ls

# # def adjust_learning_rate(optimizer, epoch):
# #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
# #     modellrnew = modellr * (0.1 ** (epoch // 50))
# #     print("lr:", modellrnew)
# #     for param_group in optimizer.param_groups:
# #         param_group['lr'] = modellrnew

    

def main():

    metric_dir = os.path.join(store_dir, 'metric_' +  time_name + '.csv')
    
    utils.mkdir(store_dir)
    
    # 创建
    with open(str(metric_dir), 'w') as f:
        fields = ['EPOCH','train_acc_last_task', 'val_acc_ave', 'val_recall_ave_pvrl', 'val_recall_ave_others']
        datawrite = csv.writer(f, delimiter=' ')
        datawrite.writerow(fields)
        
    # 创建一个记录每一次梯度的，用于观察
    metric_dir_loop = os.path.join(store_dir, 'metric_' +  time_name + '_loop.csv')
    with open(str(metric_dir_loop), 'w') as f:
        fields = ['EPOCH'] + list(range(1, args.n_inner_updates+1, 1))
        datawrite = csv.writer(f, delimiter=' ')
        datawrite.writerow(fields)      
    



    df_train, df_val, df_test_ls,train_classes, val_classes, test_classes = ori.originze_df()

    
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
    test_support_loader = DataLoader(BasicDataset(test_support_fileroots_all_task), batch_size=args.test_meta_size, shuffle=True, num_workers=0, pin_memory=True)
    test_query_loader = DataLoader(BasicDataset(test_query_fileroots_alltask), batch_size=args.test_meta_size, shuffle=False, num_workers=0, pin_memory=True)
    
    best_pred = args.best_acc
    for epoch in range(begin_epoch, 200):
        ave_acc_all_tasks, acc_last_task = train(zip(train_support_loader, train_query_loader), epoch)

        val_value = val(zip(val_support_loader, val_query_loader), epoch)
        
        
        with open(str(metric_dir), 'a+') as f:
            csv_write = csv.writer(f, delimiter=' ')
            data_row = [epoch, acc_last_task, val_value[0], val_value[2], val_value[4]]
            csv_write.writerow(data_row)
            
        with open(str(metric_dir_loop), 'a+') as f:
            csv_write = csv.writer(f, delimiter=' ')
            data_row = [epoch] + val_value[6]
            csv_write.writerow(data_row)

        is_best = val_value[0] > best_pred
        best_pred = max(best_pred, val_value[0])
        if is_best:
            torch.save(net.state_dict(), os.path.join(store_dir, 'meta_potential_model' + time_name + '.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved!')

        if epoch+1 % 10 == 0:
            test_model = deepcopy(net)
            params = deepcopy(torch.load(os.path.join(store_dir, 'meta_potential_model' + time_name + '.pth')))
            test_model.load_state_dict(params)

            dl_test(test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask)  

        
# def unpack_batch(batch):
#     # train_inputs, train_targets = batch[0]
#     # train_inputs = train_inputs.cuda()
#     # train_targets = train_targets.cuda()
#     #
#     # test_inputs, test_targets = batch[1]
#     # test_inputs = test_inputs.cuda()
#     #
#     # test_targets = test_targets.cuda()

#     device = torch.device('cuda')
#     train_inputs, train_targets = batch[0]
#     train_inputs = train_inputs.to(device=device)
#     train_targets = train_targets.to(device=device, dtype=torch.long)

#     test_inputs, test_targets = batch[1]
#     test_inputs = test_inputs.to(device=device)
#     test_targets = test_targets.to(device=device, dtype=torch.long)

#     return train_inputs, train_targets, test_inputs, test_targets    
# def train(db, epoch):
#     net.train()
#     criterion = nn.CrossEntropyLoss()
    
#     qry_acc_all_tasks = []# 所有batch tasks的query set 的acc
#     tqdm_train = tqdm(db)
#     for batch_idx, batch in enumerate(tqdm_train, 1):
#         support_x, support_y, query_x, query_y = unpack_batch(batch)
#         task_num, setsz, c_, h, w = support_x.size()
#         querysz = query_x.size(1)
        
#         # Initialize the inner optimizer to adapt the parameters to
#         # the support set.
#         # 每一个epoch都实例化一个，不用从上一个epoch开始
#         # 为什么呢
#         inner_opt = torch.optim.SGD(net.parameters(), lr=args.inner_lr)
        
#         qry_losses = []
#         qry_acc_list=[] # 存储一个batch task，query set的acc
#         meta_opt.zero_grad()
#         for i in range(task_num):
#             with higher.innerloop_ctx(net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
#                 # Optimize the likelihood of the support set by taking
#                 # gradient steps w.r.t. the model's parameters.
#                 # This adapts the model's meta-parameters to the task.
#                 # higher is able to automatically keep copies of
#                 # your network's parameters as they are being updated.
#                 for _ in range(args.n_inner_updates):
#                     spt_logits = fnet(support_x[i])
#                     spt_loss = criterion(spt_logits, support_y[i])

#                     diffopt.step(spt_loss)
#                 # The final set of adapted parameters will induce some
#                 # final loss and accuracy on the query dataset.
#                 # These will be used to update the model's meta-parameters.
#                 qry_logits = fnet(query_x[i])
#                 qry_loss = criterion(qry_logits, query_y[i])
#                 qry_losses.append(qry_loss.detach().cpu())
                
#                 acc_tmp = (qry_logits.argmax(dim=1) == query_y[i]).sum().item() / querysz
#                 acc=evl.acc_1(qry_logits.detach().cpu(), query_y[i].detach().cpu())               
#                 qry_acc_list.append(acc)
                
#                 # Update the model's meta-parameters to optimize the query
#                 # losses across all of the tasks sampled in this batch.
#                 # This unrolls through the gradient steps.
#                 qry_loss.backward()  # 为什么要在这呢
                
#         meta_opt.step()
#         qry_losses = sum(qry_losses) / task_num
#         # qry_dscs = 100. * sum(qry_dscs) / task_num # .* 和*有什么区别吗？没有吧
#         acc_ave = sum(qry_acc_list) / task_num
        
#         qry_acc_all_tasks.append(acc_ave)
        
#         tqdm_train.set_description('Training_Tasks Epoch {}, batch_idx {}, acc={:.4f}, , Loss={:.4f}'.format(epoch, batch_idx, acc_ave.item(), qry_losses))
#         if batch_idx % 5 == 0:
#             print('\n')
#             logging.info(f'step {batch_idx + 1} training: {acc_ave}!')
            
            
#     ave_qry_acc_all_tasks = np.array(qry_acc_all_tasks).mean()    
    
#     # 返回：1.所有batch tasks的query set 的acc的均值. 2.最以后一个batch task的query set的acc    
#     return ave_qry_acc_all_tasks, qry_acc_all_tasks[-1].item()

# def val(db, epoch):
#     val_net = deepcopy(net)
#     val_net.train()     
#     criterion = nn.CrossEntropyLoss()  
            
#     qry_losses = []
#     qry_acc_list=[] # 所有测试任务
#     qry_acc_inner = [0 for i in range(args.n_inner_updates)] # 任务内循环每一次梯度更新，均值
#     class_PVRL_recall_ls = []
#     class_others_recall_ls = []
    
#     tqdm_val = tqdm(db)
#     for batch_idx, batch in enumerate(tqdm_val, 1):
#         support_x, support_y, query_x, query_y = unpack_batch(batch)
#         task_num, setsz, c_, h, w = support_x.size()
#         querysz = query_x.size(1)
        
#         n_inner_iter = args.n_inner_updates
#         inner_opt = torch.optim.SGD(val_net.parameters(), lr=args.inner_lr)
#         # track_higher_grads=False,随着innerloop加深，不会增加内存使用
#         for i in range(task_num):
#             with higher.innerloop_ctx(val_net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
#                 # Optimize the likelihood of the support set by taking
#                 # gradient steps w.r.t. the model's parameters.
#                 # This adapts the model's meta-parameters to the task.
#                 for i in range(n_inner_iter):
#                     spt_logits = fnet(support_x[i])
#                     spt_loss = criterion(spt_logits, support_y[i])
#                     diffopt.step(spt_loss)
                    
#                     qry_logits = fnet(query_x[i])
#                     logits = qry_logits.detach().cpu()
#                     labels = query_y[i].detach().cpu()
#                     _, pred = torch.max(logits.data, 1)
#                     acc = accuracy_score(pred, labels)
#                     acc_1=evl.acc_1(logits, labels)
#                     acc_2 = (qry_logits.argmax(dim=1) == query_y[i]).sum().item() / querysz
                    
#                     qry_acc_inner[i] += acc
                     
                
#                 # The query loss and acc induced by these parameters.
#                 # qry_logits = fnet(query_x[i]).detach() #放到循环里，记录每一次梯度更新
#                 qry_loss = criterion(qry_logits, query_y[i])
#                 qry_losses.append(qry_loss.detach().cpu()) 
                
#                 #放到循环里，记录每一次梯度更新
#                 # acc_tmp = (qry_logits.argmax(dim=1) == query_y[i]).sum().item() / querysz
#                 # acc=evl.acc_1(qry_logits.detach().cpu(), query_y[i].detach().cpu())                
#                 # logits = qry_logits.detach().cpu()
#                 # _, pred = torch.max(logits.data, 1)
#                 # labels = query_y[i].detach().cpu()
                
#                 Recall = recall_score(pred, labels, average=None)
#                 class_PVRL_recall_ls.append(Recall[0])
#                 class_others_recall_ls.append(Recall[1])
                               
#                 qry_acc_list.append(acc)
            
#         tqdm_val.set_description('Val_Tasks Epoch {}, acc={:.4f}, queryloss {:.4f}'.format(epoch, np.mean(qry_acc_list), np.mean(np.array(qry_losses))))
            
#     del val_net
#     acc_ave = np.array(qry_acc_list).mean()
#     std = np.array(qry_acc_list).std()
    
#     ave_recall_pvrl = np.array(class_PVRL_recall_ls).mean()
#     ave_recall_others = np.array(class_others_recall_ls).mean()

#     return [acc_ave, std, ave_recall_pvrl, np.array(class_PVRL_recall_ls).std(), ave_recall_others, np.array(class_others_recall_ls).std(), list(map(lambda x:x/args.n_val_tasks,qry_acc_inner))]

# # 测试一：按照元学习方式进行测试
# # 要求：support set 和query set 不能太大，因为是整个set输入网络 
# # 就和val基本一样              
# def meta_test(db, epoch):
#     val_net = deepcopy(net)
#     val_net.train()     
#     criterion = nn.CrossEntropyLoss()  
            
#     qry_losses = []
#     qry_acc_list=[] # 所有测试任务
#     qry_acc_inner = [0 for i in range(args.n_inner_updates)] # 任务内循环每一次梯度更新，均值
#     class_PVRL_recall_ls = []
#     class_others_recall_ls = []
    
#     tqdm_val = tqdm(db)
#     for batch_idx, batch in enumerate(tqdm_val, 1):
#         support_x, support_y, query_x, query_y = unpack_batch(batch)
#         task_num, setsz, c_, h, w = support_x.size()
#         querysz = query_x.size(1)
        
#         n_inner_iter = args.n_inner_updates
#         inner_opt = torch.optim.SGD(val_net.parameters(), lr=args.inner_lr)
#         # track_higher_grads=False,随着innerloop加深，不会增加内存使用
#         for i in range(task_num):
#             with higher.innerloop_ctx(val_net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
#                 # Optimize the likelihood of the support set by taking
#                 # gradient steps w.r.t. the model's parameters.
#                 # This adapts the model's meta-parameters to the task.
#                 for i in range(n_inner_iter):
#                     spt_logits = fnet(support_x[i])
#                     spt_loss = criterion(spt_logits, support_y[i])
#                     diffopt.step(spt_loss)
                    
#                     qry_logits = fnet(query_x[i])
#                     logits = qry_logits.detach().cpu()
#                     labels = query_y[i].detach().cpu()
#                     _, pred = torch.max(logits.data, 1)
#                     acc = accuracy_score(pred, labels)
#                     acc_1=evl.acc_1(logits, labels)
#                     acc_2 = (qry_logits.argmax(dim=1) == query_y[i]).sum().item() / querysz
                    
#                     qry_acc_inner[i] += acc
                     
                
#                 # The query loss and acc induced by these parameters.
#                 # qry_logits = fnet(query_x[i]).detach() #放到循环里，记录每一次梯度更新
#                 qry_loss = criterion(qry_logits, query_y[i])
#                 qry_losses.append(qry_loss.detach().cpu()) 
                
#                 #放到循环里，记录每一次梯度更新
#                 # acc_tmp = (qry_logits.argmax(dim=1) == query_y[i]).sum().item() / querysz
#                 # acc=evl.acc_1(qry_logits.detach().cpu(), query_y[i].detach().cpu())                
#                 # logits = qry_logits.detach().cpu()
#                 # _, pred = torch.max(logits.data, 1)
#                 # labels = query_y[i].detach().cpu()
                
#                 Recall = recall_score(pred, labels, average=None)
#                 class_PVRL_recall_ls.append(Recall[0])
#                 class_others_recall_ls.append(Recall[1])
                               
#                 qry_acc_list.append(acc)
            
#         tqdm_val.set_description('Val_Tasks Epoch {}, acc={:.4f}, queryloss {:.4f}'.format(epoch, np.mean(qry_acc_list), np.mean(np.array(qry_losses))))
            
#     del val_net
#     acc_ave = np.array(qry_acc_list).mean()
#     std = np.array(qry_acc_list).std()
    
#     ave_recall_pvrl = np.array(class_PVRL_recall_ls).mean()
#     ave_recall_others = np.array(class_others_recall_ls).mean()


    
    
#     return [acc_ave, std, ave_recall_pvrl, np.array(class_PVRL_recall_ls).std(), ave_recall_others, np.array(class_others_recall_ls).std(), list(map(lambda x:x/args.n_val_tasks,qry_acc_inner))]




# 测试二：按照深度学习进行测试
# 需要将S和Q重新组batch,S作为train data, Q 作为test data
def dl_test(sppport_all_task, query_all_task, final_test_task,meta_epoch, model):

    # 创建一个记录测试任务的
    metric_dir = os.path.join(store_dir, 'metric_' +  time_name + '_test.csv')
    with open(str(metric_dir), 'w') as f:
        fields = ['task_idx', 'epoch', 'loss','acc', 'precision','recall','f1','best_acc']
        datawrite = csv.writer(f, delimiter=' ')
        datawrite.writerow(fields)  

    # model = deepcopy(net)

    dl_ob = dl_comm(model, device)

    task_num = sppport_all_task.size(0)
    for task_idx in range(task_num):
        train_data_list = sppport_all_task[task_idx]
        
        val_data_list = query_all_task[task_idx]
        test_data_list = final_test_task[task_idx]

        train_set = DLdataset(train_data_list, mode='train')
        val_set = DLdataset(val_data_list, mode='val')
        test_set = DLdataset(test_data_list, mode='test')



        train_loader = DataLoader(train_set, shuffle=True, batch_size = 16, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size = 4, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=True, batch_size = 1, num_workers=4, pin_memory=True)

        best_pred = 0
        for epoch in range(1, 100 + 1):
            dl_ob.adjust_learning_rate(epoch)
            dl_ob.train(train_loader, epoch)
            val_value = dl_ob.val(val_loader, epoch)

            with open(str(metric_dir), 'a+') as f:
                csv_write = csv.writer(f, delimiter=' ')
                data_row = [task_idx, epoch] + val_value
                data_row.append(best_pred)
                csv_write.writerow(data_row)

            is_best = val_value[1] > best_pred
            best_pred = max(best_pred, val_value[1])
            if is_best:
                torch.save(dl_ob.model.state_dict(),os.path.join(store_dir, 'best_model' + time_name + '.pth'))
                logging.info(f'Checkpoint {epoch + 1} saved!')



        params = deepcopy(torch.load(os.path.join(store_dir, 'max_dsc_' + time_name + '.pth')))
        model.load_state_dict(params)     
        test_values = dl_ob.test(test_loader, epoch)

        with open(str(metric_dir), 'a+') as f:
            csv_write = csv.writer(f, delimiter=' ')
            data_row = ['final_test', 'nihao'] + test_values
            # data_row.append(best_pred)
            csv_write.writerow(data_row)        

   

               

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
    store_dir =str(os.path.join(project_path, 'maml', args.save_path, args.prefix, args.net, time_name))

    main()