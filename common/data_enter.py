
import os
from torch.utils.data import  DataLoader
import time

from common.originize_df import originze_df_maml_four
from common.lymph.originize_ly_df import ly_originze_df,originze_source_data_pre
from common.lymph.dataloader_ly import LY_MetaDataset, LY_dataset
from common.imagenet.originize_imagenet import originze_imagenet
from common.build_tasks import Train_task,Meta_Test_task
from common.dataloader import *


def oct_data(args):
# 组织数据
    sourcee_class = ["normal", "acute CSCR", "acute RAO", "acute RVO", "acute VKH",  "dAMD", "macular-off RRD", "mCNV", "MTM", "nAMD", "nPDR", "PCV", "PDR"]
    target_classes = ['PIC', 'PVRL',  'RP'] 
    df_train, df_test_dc = originze_df_maml_four(args.project_path, sourcee_class, target_classes)

    # # 直接使用
    # df_test_s = pd.read_csv(os.path.join(args.project_path, 'data/linux/all_device/6_2_2_10/four_classes/s_q_t/s.csv'))
    # df_test_q = pd.read_csv(os.path.join(args.project_path, 'data/linux/all_device/6_2_2_10/four_classes/s_q_t/q.csv'))
    # df_test_t = pd.read_csv(os.path.join(args.project_path, 'data/linux/all_device/6_2_2_10/four_classes/s_q_t/t.csv'))
    # df_test_ls = [df_test_s, df_test_q, df_test_t]

    # 训练集，训练任务
    train_support_fileroots_alltask, train_query_fileroots_alltask = [], []
    for each_task in range(args.n_train_tasks):  # num_train_task 训练任务的总数
        task = Train_task(args,df_train,each_task, mode='train')
        train_support_fileroots_alltask.append(task.support_roots)
        train_query_fileroots_alltask.append(task.query_roots)

    # 测试集，测试任务
    test_shot = args.test_k_shot  # ，每个测试任务，每个类别包含的训练样本的数量
    test_ways = args.n_way  # 
    test_query = args.test_k_qry  # ， 每个测试任务，每个类别包含的测试样本的数量
    # support相当于验证集，可用于算法本身的优化和调整，但不能作为最终模型好坏的评价数据
    test_query_fileroots_alltask, test_support_fileroots_all_task = [], []
    final_test_alltask = []#用于最终测试
    # for each_task in range(args.n_test_tasks):  # 测试任务总数
    for source_class_name in df_test_dc:  # 测试任务总数
        test_task = Meta_Test_task(test_shot, test_query, df_test_dc[source_class_name], source_class_name)
        test_query_fileroots_alltask.append(test_task.query_roots)
        test_support_fileroots_all_task.append(test_task.support_roots)
        final_test_alltask.append(test_task.test_roots)
        
    
    test_data_ls = [test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask]

    # DataLoader，train_task S 和Q   
    train_support_loader = DataLoader(BasicDataset(train_support_fileroots_alltask, resize=args.resize), batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)
    train_query_loader = DataLoader(BasicDataset(train_query_fileroots_alltask, resize=args.resize), batch_size=args.meta_size, shuffle=True, num_workers=0, pin_memory=True)
    train_data_ls = [train_support_loader, train_query_loader]

    return test_data_ls, train_data_ls

def lymph_data(args):
    # 组织数据
    df_train, df_test_dc = ly_originze_df(args)


    # 测试集，测试任务
    # query set相当于验证集，可用于算法本身的优化和调整，但不能作为最终模型好坏的评价数据
    test_query_fileroots_alltask, test_support_fileroots_all_task = [], []
    final_test_alltask = []#用于最终测试
    # for each_task in range(args.n_test_tasks):  # 测试任务总数
    for key, df_test in df_test_dc.items():  # 测试任务总数
        test_task = Meta_Test_task(args, df_test, test_class='lymph')
        # test_query_fileroots_alltask.append(test_task.query_roots)
        # test_support_fileroots_all_task.append(test_task.support_roots)
        # # final_test_alltask.append(test_task.test_roots)
        # test_support_fileroots_all_task.append(LY_dataset(args, test_task.support_roots, mode='train'))
        # test_query_fileroots_alltask.append(LY_dataset(args, test_task.query_roots, mode='val'))
        # final_test_alltask.append(LY_dataset(args, test_task.test_roots, mode='test'))

        train_set = LY_dataset(args, test_task.support_roots, mode='train')
        val_set = LY_dataset(args, test_task.query_roots, mode='val')
        test_set = LY_dataset(args, test_task.test_roots, mode='test')

        train_loader = DataLoader(train_set, shuffle=True, batch_size = args.batch_size_train, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, batch_size = args.batch_size_val, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=False, batch_size = args.batch_size_test, num_workers=args.num_workers, pin_memory=True)

        test_support_fileroots_all_task.append(train_loader)
        test_query_fileroots_alltask.append(val_loader)
        final_test_alltask.append(test_loader)
        
    test_data_ls = [test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask]


    # 构建训练任务
    # 训练集，训练任务
    train_support_fileroots_alltask, train_query_fileroots_alltask = [], []
    for each_task in range(args.n_train_tasks):  # num_train_task 训练任务的总数
        task = Train_task(args,df_train,each_task, mode='train')
        train_support_fileroots_alltask.append(task.support_roots)
        train_query_fileroots_alltask.append(task.query_roots)

    # DataLoader，train_task S 和Q   
    train_support_loader = DataLoader(LY_MetaDataset(args, train_support_fileroots_alltask), batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)
    train_query_loader = DataLoader(LY_MetaDataset(args, train_query_fileroots_alltask), batch_size=args.meta_size, shuffle=True, num_workers=0, pin_memory=True)
    train_data_ls = [train_support_loader, train_query_loader]

    return test_data_ls, train_data_ls

def source_data_pre(args):

    df_train, df_test_dc = originze_source_data_pre()

    # 测试集，测试任务
    # query相当于验证集，可用于算法本身的优化和调整，但不能作为最终模型好坏的评价数据
    test_query_fileroots_alltask, test_support_fileroots_all_task = [], []
    final_test_alltask = []#用于最终测试
    # for each_task in range(args.n_test_tasks):  # 测试任务总数
    for key, df_test in df_test_dc.items():  # 测试任务总数
        test_task = Meta_Test_task(args, df_test, test_class='lymph')
        test_query_fileroots_alltask.append(test_task.query_roots)
        test_support_fileroots_all_task.append(test_task.support_roots)
        final_test_alltask.append(test_task.test_roots)
    test_data_ls = [test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask]

    train_data_ls = []

    return test_data_ls, train_data_ls

def imagenet_exe(args):
    df_train, df_test_dc = originze_source_data_pre()

    # 测试集，测试任务
    # query相当于验证集，可用于算法本身的优化和调整，但不能作为最终模型好坏的评价数据
    test_query_fileroots_alltask, test_support_fileroots_all_task = [], []
    final_test_alltask = []#用于最终测试
    # for each_task in range(args.n_test_tasks):  # 测试任务总数
    for key, df_test in df_test_dc.items():  # 测试任务总数
        test_task = Meta_Test_task(args, df_test, test_class='lymph')
        test_query_fileroots_alltask.append(test_task.query_roots)
        test_support_fileroots_all_task.append(test_task.support_roots)
        final_test_alltask.append(test_task.test_roots)
    test_data_ls = [test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask]

    train_data_ls = []

    return test_data_ls, train_data_ls


