import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')
import numpy as np
import torch
import time
from collections import OrderedDict
from tqdm import tqdm
import os
import csv
from torch.utils.tensorboard import SummaryWriter
from common.meta.meta_comm import MAML
import common.utils as utils
import dl.dl_func as dlf

def maml_enter(args, train_data, test_data):

    # store_dir_meta_test =str(os.path.join(project_path,args.save_path, args.alg.lower(), args.net, time_name, 'meta_test'))
    store_dir_meta_test = args.store_dir
    # utils.mkdir(store_dir_meta_test)
    store_meta_pth = str(os.path.join(args.store_dir, 'save_meta_pth'))
    utils.mkdir(store_meta_pth)
    
    # 创建一个记录meta_train, 用于观察
    metric_dir_meta_train = os.path.join(args.store_dir, 'metric_meta_train' + '.csv')
    with open(str(metric_dir_meta_train), 'w') as f:
        fields = ['Meta_Epoch', 'Loss', 'ACC', 'Lr']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)

    # 创建一个记录meta_test的query set的acc
    metric_dir_meta_test = os.path.join(store_dir_meta_test, 'metric_meta_test' + '.csv')
    with open(str(metric_dir_meta_test), 'w') as f:
        fields = ['Meta_Epoch'] + list(range(1, args.n_epoch + 1, 1)) + ['best_val_acc', 'best_val_epoch', 'best_Q_acc', 'best_meta_epoch', 'best_Q_acc_1', 'best_meta_epoch_1', 'final_test_acc']#记录内循环每次的acc
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)

    model = MAML(args)
    best_pred_for_final = 0.0 #元测试中，测试任务 querry set上最好的值，使用所有任务，相同epoch的均值
    best_epoch_for_final = 0

    # 另外一种计算方式， 使用所有任务，每个任务最好的val_acc
    best_val = 0.0
    best_epoch = 0

    tensor_writer = SummaryWriter('{}/tensorboard_log/meta_train'.format(args.store_dir))

    for epoch in range(args.n_meta_epoch):
        res, meta_test_Q_values, meta_final_ave_tasks= run_epoch(epoch, args, model, zip(train_data[0], train_data[1]), test_data)    

        # 使用所有任务，相同epoch的均值
        best_meta_test_value =  max(meta_test_Q_values)            
        is_best_for_final =best_meta_test_value > best_pred_for_final
        best_pred_for_final = max(best_pred_for_final, best_meta_test_value)
        if is_best_for_final :
            best_epoch_for_final = epoch  

        # 使用所有任务，最好的val_acc
        is_best_2 = meta_final_ave_tasks[-2] > best_val
        best_val = max(meta_final_ave_tasks[-2], best_val)
        if is_best_2:
            best_epoch = epoch

        with open(str(metric_dir_meta_test), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            
            best_val_acc = max(meta_test_Q_values)
            best_index = meta_test_Q_values.index(best_val_acc)
           
            data_row = [epoch] + meta_test_Q_values + [best_val_acc, best_index+1,best_pred_for_final,best_epoch_for_final, best_val,best_epoch, meta_final_ave_tasks[1]]
            csv_write.writerow(data_row)

        current_outer_lr =  model.outer_optimizer.state_dict()['param_groups'][0]['lr']
        with open(str(metric_dir_meta_train), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            data_row = [res['epoch'], res['train_loss'], res['train_acc'], current_outer_lr]
            csv_write.writerow(data_row)

        tensor_writer.add_scalar('meta_train_loss', res['train_loss'], epoch)
        tensor_writer.add_scalar('meta_train_acc',  res['train_acc'], epoch)
        tensor_writer.add_scalar('meta_learn_rate', current_outer_lr, epoch)
        
        
        torch.cuda.empty_cache()
        # 默认不执行
        if args.lr_sched:
            model.lr_sched()

def run_epoch(epoch, args, model, train_loader, test_ls):
    res = OrderedDict()
    print('Epoch {}'.format(epoch))

    ave_acc_all_tasks, acc_last_task, train_loss= model.train(train_loader, epoch)

    res['epoch'] = epoch
    res['train_acc'] = ave_acc_all_tasks
    res['train_loss'] = train_loss


    # 保存每一个epoch的元参数
    meta_epoch_model = os.path.join(args.store_dir, 'save_meta_pth',f'meta_epoch_{epoch}.pth')
    torch.save(model.network.state_dict(), meta_epoch_model)

    meta_test_Q_values, meta_final_ave_tasks = dlf.trainer(args, test_ls[0], test_ls[1], test_ls[2], epoch)

    return res, meta_test_Q_values, meta_final_ave_tasks