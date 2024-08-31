import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split
import common.utils as utils


# # 小函数 
# def getdf(dframe, source_cls, label_dict):

#     df_for_train = pd.DataFrame()
#     add_classID(dframe, label_dict)
#     df_arg = dframe[dframe['Class'] == 'micro']
#     source_cls = dframe[dframe['Class'] == source_cls]
    

#     df_for_train = pd.concat([df_micro, df_common],ignore_index=True)

#     return df_for_train

# 小函数，增加label列
def add_classID(pdframe, dict_labels):

    label_list = []
    for class_name in pdframe['Class']:
        if class_name in dict_labels:
            label_list.append(dict_labels[class_name])
        else:
            label_list.append('others')
    
    pdframe['Label'] = label_list
    return pdframe


# 仅构造一个测试任务
def ly_originze_df_one(project_path, datatype):

    source_classes = []
    target_classes = []
    # target_df
    if datatype == 'lymph-4x' or datatype == 'lymph-10x':
        label_dict = {'micro':1, "common":0}
        target_classes = ['micro', 'common']

    elif datatype == 'thyroid-4x' or datatype == 'thyroid-10x': 
        label_dict = {'thyroid-micro':1,  'thyroid-normal':0}   
        target_classes = {'thyroid-micro', 'thyroid-normal'}

    elif datatype == 'ffpe-4x' or datatype == 'ffpe-10x': 
        label_dict = {'ffpe_micro':1,  'ffpe_normal':0}   
        target_classes = {'ffpe_micro', 'ffpe_normal'}

    if datatype == 'lymph-4x':
        df_train =  pd.read_csv(os.path.join(project_path, 'data/lymph/9-11/6_2_2/train_4x.csv'))
        df_val = pd.read_csv(os.path.join(project_path, 'data/lymph/9-11/6_2_2/val_4x.csv'))
        df_test = pd.read_csv(os.path.join(project_path, 'data/lymph/9-11/6_2_2/test_4x.csv'))
    elif datatype == 'lymph-10x':
        df_train =  pd.read_csv(os.path.join(project_path, 'data/lymph/9-11/6_2_2/train_10x.csv'))
        df_val = pd.read_csv(os.path.join(project_path, 'data/lymph/9-11/6_2_2/val_10x.csv'))
        df_test = pd.read_csv(os.path.join(project_path, 'data/lymph/9-11/6_2_2/test_10x.csv'))

    elif datatype == 'thyroid-4x':
        df_train =  pd.read_csv(os.path.join(project_path, 'data/lymph/thyroid/split_by_node/train_4x.csv'))
        df_val = pd.read_csv(os.path.join(project_path, 'data/lymph/thyroid/split_by_node/val_4x.csv'))
        df_test = pd.read_csv(os.path.join(project_path, 'data/lymph/thyroid/split_by_node/test_4x.csv'))

    elif datatype == 'ffpe-4x':
        df_train =  pd.read_csv(os.path.join(project_path, 'data/lymph/breast/FFPE/split_by_node/4x/24_8_8/balance/train_4x.csv'))
        df_val = pd.read_csv(os.path.join(project_path, 'data/lymph/breast/FFPE/split_by_node/4x/24_8_8/balance/test_4x.csv')) # 没有写错，不要更改
        df_test = pd.read_csv(os.path.join(project_path, 'data/lymph/breast/FFPE/split_by_node/4x/24_8_8/balance/val_4x.csv'))
    else:
       raise ValueError('Not implemented ly_originze_df') 

    df_meta_test_S = pd.DataFrame()
    df_meta_test_Q = pd.DataFrame()
    df_final_test = pd.DataFrame()

    # 获取目标类的df
    for class_name in target_classes:
        df_class_train = df_train[df_train['Class'] == class_name]
        df_class_val = df_val[df_val['Class'] == class_name]
        df_class_test = df_test[df_test['Class'] == class_name]
        
        df_meta_test_S = pd.concat([df_meta_test_S, df_class_train], ignore_index=True)
        df_meta_test_Q = pd.concat([df_meta_test_Q, df_class_val], ignore_index=True)
        df_final_test = pd.concat([df_final_test, df_class_test], ignore_index=True)

    # target_classes的df增加label列
    store_path = f'/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/{datatype}/meta_tasks/final_test_tasks/'
    test_d = {}
    test_S = add_classID(df_meta_test_S, label_dict)
    test_Q = add_classID(df_meta_test_Q, label_dict)
    final_test = add_classID(df_final_test, label_dict)

    # 测试任务需要是一个列表，因为不止一个任务
    test_d[0] = [test_S, test_Q, final_test]


    test_S.to_csv(store_path + '_s.csv')
    test_Q.to_csv(store_path + '_q.csv')
    final_test.to_csv(store_path + '_f.csv')


    df_meta_train = pd.read_csv(os.path.join(project_path, 'data/lymph/source_data/tissue_and_source.csv'))
    source_label_dict = {'ryzh': 0, 'liver': 1, 'gzfb': 2, 'xwzhblyb': 3, 'rg': 4, 'parathyroid': 5, 'wnm-wt': 6, 'hwj': 7, 'xy': 8, 'zqgsp': 9, 'sgs': 10, 'colon': 11, 'gyx': 12, 'pgbysp': 13, 'wbm-ym': 14, 'xwzh': 15, 'nong': 16, 'bp': 17, 'zf': 18, 'xg': 19, 'hs': 20, 'smoothmuscle': 21, 'sxgsz': 22, 'adrenalcortex': 23, 'brain': 24, 'intestinalvillus': 25, 'lung': 26, 'mammarygland': 27, 'pancreas': 28, 'prostate': 29, 'smallintestinalgland': 30, 'spleen': 31, 'thyroid': 32, 'tonsil': 33}
    
    df_meta_train = add_classID(df_meta_train, source_label_dict)
    

    return df_meta_train, test_d


# 构造多个测试任务
def ly_originze_df(args):

    source_classes = []
    target_classes = []
    # target_df
    if args.datatype == 'lymph-4x' or args.datatype == 'lymph-10x':
        label_dict = {'micro':1, "common":0}
        target_classes = ['micro', 'common']

    elif args.datatype == 'thyroid-4x' or args.datatype == 'thyroid-10x': 
        label_dict = {'thyroid-micro':1,  'thyroid-normal':0}   
        target_classes = ['thyroid-micro', 'thyroid-normal']

    elif args.datatype == 'background_10x':
        label_dict = {1:1,  0:0}
        target_classes = [1, 0]

    elif args.datatype == 'ffpe-4x' or args.datatype == 'ffpe-10x' or 'multi_centers_ffpe' in args.datatype: 
        label_dict = {'ffpe_micro':1,  'ffpe_normal':0}   
        target_classes = ['ffpe_micro', 'ffpe_normal']
    elif args.datatype == 'bf-4x' or args.datatype == 'bf-10x' or 'multi_centers_bf' in args.datatype: 
        label_dict = {'bf_micro':1,  'bf_normal':0}   
        target_classes = ['bf_micro', 'bf_normal']
    elif 'camely_all' in args.datatype:
        label_dict = {'ffpe_macro':1, 'ffpe_itc':1, 'ffpe_micro':1,  'ffpe_normal':0}
        if args.predict_class != []:
            target_classes = args.predict_class
        else:   
            target_classes = ['ffpe_itc', 'ffpe_micro', 'ffpe_normal']
    elif 'camely_17' in args.datatype:
        label_dict = {'ffpe_itc':1, 'ffpe_micro':1,  'ffpe_normal':0}  
        if args.predict_class != []:
            target_classes = args.predict_class
        else:
            target_classes = ['ffpe_itc', 'ffpe_micro', 'ffpe_normal']

    elif 'camely_16' in args.datatype: 
        label_dict = {'ffpe_macro':1, 'ffpe_micro':1,  'ffpe_normal':0}
        if args.predict_class != []:
            target_classes = args.predict_class
        else:
            target_classes = ['ffpe_micro', 'ffpe_normal']
    else:
        raise ValueError(f'Not implemented label ly_originze_df args.datatype {args.datatype}') 

    test_d = {}
    store_path = f'/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/meta_tasks/{args.datatype}/final_test_tasks/'
    utils.mkdir(store_path)
    for i in range(args.n_test_tasks):

        if args.datatype == 'lymph-4x':
            df_train =  pd.read_csv(os.path.join(args.project_path, 'data/lymph/9-11/6_2_2/train_4x.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, 'data/lymph/9-11/6_2_2/val_4x.csv'))
            df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/9-11/6_2_2/test_4x.csv'))
        elif args.datatype == 'lymph-10x':
            df_train =  pd.read_csv(os.path.join(args.project_path, 'data/lymph/9-11/6_2_2/train_4x.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, 'data/lymph/9-11/6_2_2/val_4x.csv'))
            df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/9-11/6_2_2/test_4x.csv'))

        elif args.datatype == 'background_10x':
            df_train =  pd.read_csv(os.path.join(args.project_path, 'data/lymph/background/10x/all.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, 'data/lymph/background/10x/val.csv'))
            df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/background/10x/all.csv'))

        elif args.datatype == 'thyroid_4x':
            df_train =  pd.read_csv(os.path.join(args.project_path, 'data/lymph/thyroid/split_by_node/train_4x.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, 'data/lymph/thyroid/split_by_node/val_4x.csv'))
            df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/thyroid/split_by_node/test_4x.csv'))
        
        elif args.datatype == 'thyroid_10x':
            df_train =  pd.read_csv(os.path.join(args.project_path, 'data/lymph/thyroid/split_by_node/train_4x.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, 'data/lymph/thyroid/split_by_node/val_4x.csv'))
            df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/thyroid/split_by_node/test_4x.csv'))

        elif args.datatype == 'ffpe-4x':
            df_train =  pd.read_csv(os.path.join(args.project_path, f'data/lymph/breast/FFPE/split_by_node/4x/24_8_8/balance/multi_tasks/t_{i}_s.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, f'data/lymph/breast/FFPE/split_by_node/4x/24_8_8/balance/multi_tasks/t_{i}_q.csv'))
            df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/breast/FFPE/split_by_node/4x/24_8_8/balance/test_4x.csv')) 
        elif args.datatype == 'ffpe-10x':
            df_train =  pd.read_csv(os.path.join(args.project_path, f'data/lymph/breast/FFPE/split_by_node/10x/balance/multi_tasks/t_{i}_s.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, f'data/lymph/breast/FFPE/split_by_node/10x/balance/multi_tasks/t_{i}_q.csv'))
            df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/breast/FFPE/split_by_node/10x/balance/test_10x.csv')) 
        elif args.datatype == 'bf-4x':
            df_train = pd.read_csv(os.path.join(args.project_path, 'data/lymph/breast/BF/split_by_node/4x/test_4x.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, 'data/lymph/breast/BF/split_by_node/4x/test_4x.csv'))
            df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/breast/BF/split_by_node/4x/test_4x.csv'))
        elif args.datatype == 'bf-10x':
            df_train = pd.read_csv(os.path.join(args.project_path, 'data/lymph/breast/BF/split_by_node/10x/test_10x.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, 'data/lymph/breast/BF/split_by_node/10x/test_10x.csv'))
            df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/breast/BF/split_by_node/10x/test_10x.csv'))

        elif args.datatype == 'multi_centers_ffpe_10x':
            df_train =  pd.read_csv(os.path.join(args.project_path, f'data/lymph/multi_centers/all_centers/multi_tasks/ffpe_10x/t_{i}_s.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, f'data/lymph/multi_centers/all_centers/multi_tasks/ffpe_10x/t_{i}_q.csv'))

            if args.multi_centers_test == 'jingxia':
                df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/multi_centers/captions/patch_use/10x.csv'))
                df_test_micro = df_test[df_test['Class'] == 'ffpe_micro']
                df_test_normal = pd.read_csv(os.path.join(args.project_path,'data/lymph/multi_centers/captions/patch_use/all_centers/ffpe-10x/ffpe-10x-normal.csv'))
                df_test = pd.concat([df_test_micro, df_test_normal], ignore_index=True)

            elif  args.multi_centers_test == 'throid':
                # df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/thyroid/10x/10x.csv'))
                # df_test_micro = df_test[df_test['Class'] == 'ffpe_micro']
                df_test_micro =  pd.read_csv(os.path.join(args.project_path,'data/lymph/thyroid/10x/10x-micro.csv'))
                df_test_normal = pd.read_csv(os.path.join(args.project_path,'data/lymph/thyroid/10x/10x-normal.csv'))
                df_test = pd.concat([df_test_micro, df_test_normal], ignore_index=True)
            elif  args.multi_centers_test == 'crc':
                # df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/CRC-LN/10x/balance/micro_10x_balance.csv'))
            
                # df_test_micro = df_test[df_test['Class'] == 'ffpe_micro']
                df_test_micro = pd.read_csv(os.path.join(args.project_path,'data/lymph/CRC-LN/10x/balance/micro_10x.csv'))
                df_test_normal = pd.read_csv(os.path.join(args.project_path,'data/lymph/CRC-LN/10x/balance/normal_10x.csv'))
                df_test = pd.concat([df_test_micro, df_test_normal], ignore_index=True)
            
            else:
                df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/multi_centers/all_centers/multi_tasks/ffpe_10x/test.csv'))
        elif args.datatype == 'multi_centers_ffpe_4x':
            df_train =  pd.read_csv(os.path.join(args.project_path, f'data/lymph/multi_centers/all_centers/multi_tasks/ffpe_4x/t_{i}_s.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, f'data/lymph/multi_centers/all_centers/multi_tasks/ffpe_4x/t_{i}_q.csv'))
            if args.multi_centers_test == 'jingxia':
                df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/multi_centers/captions/patch_use/4x.csv'))
                
                df_test_micro = df_test[df_test['Class'] == 'ffpe_micro']
                df_test_normal = pd.read_csv(os.path.join(args.project_path,'data/lymph/multi_centers/captions/patch_use/all_centers/ffpe-4x/ffpe-4x-normal.csv'))
                df_test = pd.concat([df_test_micro, df_test_normal], ignore_index=True)
            
            elif args.multi_centers_test == 'throid':
                df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/thyroid/4x/4x.csv'))
            elif args.multi_centers_test == 'crc':
                df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/CRC-LN/4x/balance/micro_4x_balance.csv'))

            else:
                df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/multi_centers/all_centers/multi_tasks/ffpe_4x/test.csv'))

        elif args.datatype == 'multi_centers_bf_10x':
            df_train =  pd.read_csv(os.path.join(args.project_path, f'data/lymph/multi_centers/all_centers/multi_tasks/bf_10x/t_{i}_s.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, f'data/lymph/multi_centers/all_centers/multi_tasks/bf_10x/t_{i}_q.csv'))
            if args.multi_centers_test == 'jingxia':
                df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/multi_centers/captions/patch_use/10x.csv'))
                df_test_micro = df_test[df_test['Class'] == 'bf_micro']
                df_test_normal = pd.read_csv(os.path.join(args.project_path,'data/lymph/multi_centers/captions/patch_use/all_centers/bf-10x/bf-10x-normal.csv'))
                df_test = pd.concat([df_test_micro, df_test_normal], ignore_index=True)
            
            else:
                df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/multi_centers/all_centers/multi_tasks/bf_10x/test.csv'))

        elif args.datatype == 'multi_centers_bf_4x':
            df_train =  pd.read_csv(os.path.join(args.project_path, f'data/lymph/multi_centers/all_centers/multi_tasks/bf_4x/t_{i}_s.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, f'data/lymph/multi_centers/all_centers/multi_tasks/bf_4x/t_{i}_q.csv'))
            if args.multi_centers_test == 'jingxia':
                df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/multi_centers/captions/patch_use/4x.csv'))
                df_test_micro = df_test[df_test['Class'] == 'bf_micro']
                df_test_normal = pd.read_csv(os.path.join(args.project_path,'data/lymph/multi_centers/captions/patch_use/all_centers/bf-4x/bf-4x-normal.csv'))
                df_test = pd.concat([df_test_micro, df_test_normal], ignore_index=True)
                
            else:
                df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/multi_centers/all_centers/multi_tasks/bf_4x/test.csv'))

        elif args.datatype == 'camely_17_4x':
            df_train =  pd.read_csv(os.path.join(args.project_path, f'data/lymph/camely/camely_17/split_by_node/4x/balance/multi_tasks/t_{i}_s.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, f'data/lymph/camely/camely_17/split_by_node/4x/balance/multi_tasks/t_{i}_q.csv'))
            df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/camely/camely_17/split_by_node/4x/balance/test_4x.csv'))
        
        elif args.datatype == 'camely_17_10x':
            df_train =  pd.read_csv(os.path.join(args.project_path, f'data/lymph/camely/camely_17/split_by_node/10x/balance/multi_tasks/t_{i}_s.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, f'data/lymph/camely/camely_17/split_by_node/10x/balance/multi_tasks/t_{i}_q.csv'))
            df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/camely/camely_17/split_by_node/10x/balance/test_10x.csv'))

        elif args.datatype == 'camely_all_4x':
            df_train =  pd.read_csv(os.path.join(args.project_path, f'data/lymph/camely/camely_all/multi_tasks/4x/t_{i}_s.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, f'data/lymph/camely/camely_all/multi_tasks/4x/t_{i}_q.csv'))
            df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/camely/camely_all/multi_tasks/4x/test.csv'))
            if target_classes[0] == 'ffpe_itc' and 'predict' in args.alg and args.predict_class != []:
                # 均衡normal
                df_test = balance_itc(df_test, 0.08)


        elif args.datatype == 'camely_all_10x':
            df_train =  pd.read_csv(os.path.join(args.project_path, f'data/lymph/camely/camely_all/multi_tasks/10x/t_{i}_s.csv'))
            df_val = pd.read_csv(os.path.join(args.project_path, f'data/lymph/camely/camely_all/multi_tasks/10x/t_{i}_q.csv'))
            df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/camely/camely_all/multi_tasks/10x/test.csv'))
            if target_classes[0] == 'ffpe_itc' and 'predict' in args.alg and args.predict_class != []:
                # 均衡normal
                df_test = balance_itc(df_test, 0.06)




        else:
            raise ValueError(f'Not implemented ly_originze_df args.datatype {args.datatype}') 

        df_meta_test_S = pd.DataFrame()
        df_meta_test_Q = pd.DataFrame()
        df_final_test = pd.DataFrame()

        # 获取目标类的df
        for class_name in target_classes:
            df_class_train = df_train[df_train['Class'] == class_name]
            df_class_val = df_val[df_val['Class'] == class_name]
            df_class_test = df_test[df_test['Class'] == class_name]
            
            df_meta_test_S = pd.concat([df_meta_test_S, df_class_train], ignore_index=True)
            df_meta_test_Q = pd.concat([df_meta_test_Q, df_class_val], ignore_index=True)
            df_final_test = pd.concat([df_final_test, df_class_test], ignore_index=True)

        # target_classes的df增加label列
        
        test_S = add_classID(df_meta_test_S, label_dict)
        test_Q = add_classID(df_meta_test_Q, label_dict)
        final_test = add_classID(df_final_test, label_dict)

        # 测试任务需要是一个列表，因为不止一个任务
        test_d[i] = [test_S, test_Q, final_test]


        test_S.to_csv(store_path + f't_{i}_s.csv', index=False)
        test_Q.to_csv(store_path + f't_{i}_q.csv',index=False)
        final_test.to_csv(store_path + f't_{i}_f.csv',index=False)


    df_meta_train = pd.read_csv(os.path.join(args.project_path, 'data/lymph/source_data/tissue_and_source.csv'))
    source_label_dict = {'ryzh': 0, 'liver': 1, 'gzfb': 2, 'xwzhblyb': 3, 'rg': 4, 'parathyroid': 5, 'wnm-wt': 6, 'hwj': 7, 'xy': 8, 'zqgsp': 9, 'sgs': 10, 'colon': 11, 'gyx': 12, 'pgbysp': 13, 'wbm-ym': 14, 'xwzh': 15, 'nong': 16, 'bp': 17, 'zf': 18, 'xg': 19, 'hs': 20, 'smoothmuscle': 21, 'sxgsz': 22, 'adrenalcortex': 23, 'brain': 24, 'intestinalvillus': 25, 'lung': 26, 'mammarygland': 27, 'pancreas': 28, 'prostate': 29, 'smallintestinalgland': 30, 'spleen': 31, 'thyroid': 32, 'tonsil': 33}
    
    df_meta_train = add_classID(df_meta_train, source_label_dict)
    

    return df_meta_train, test_d

def get_meta_train():
    df_meta_train = pd.DataFrame()
    return df_meta_train

def originze_source_data_pre():
    project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/'
    df_source_data = pd.read_csv(os.path.join(project_path,'data/lymph/source_data/source_classes.csv'))

    source_label_dict = {'ryzh': 0, 'liver': 1, 'gzfb': 2, 'xwzhblyb': 3, 'rg': 4, 'parathyroid': 5, 'wnm-wt': 6, 'hwj': 7, 'xy': 8, 'zqgsp': 9, 'sgs': 10, 'colon': 11, 'gyx': 12, 'pgbysp': 13, 'wbm-ym': 14, 'xwzh': 15, 'nong': 16, 'bp': 17, 'zf': 18, 'xg': 19, 'hs': 20, 'smoothmuscle': 21, 'sxgsz': 22, 'adrenalcortex': 23, 'brain': 24, 'intestinalvillus': 25, 'lung': 26, 'mammarygland': 27, 'pancreas': 28, 'prostate': 29, 'smallintestinalgland': 30, 'spleen': 31, 'thyroid': 32, 'tonsil': 33}

    df_source_data = add_classID(df_source_data, source_label_dict)
    test_d = {}
    # 将数据集按照 8:2 划分成训练集和测试集
    train_df, test_df = train_test_split(df_source_data, test_size=0.2, random_state=42)

    test_d[0] = [train_df, test_df,test_df]

    df_meta_train = pd.DataFrame()

    return df_meta_train, test_d


# 测试itc是均衡normla
def balance_itc(dframe,frac):
    df_normal = dframe[dframe['Class'] == 'ffpe_normal']
    df_itc = dframe[dframe['Class'] == 'ffpe_itc']

    df_balance= pd.DataFrame()
    nor_all_nodes = df_normal['Node'].unique()
    for node in nor_all_nodes:
        df_node = df_normal[df_normal['Node'] == node]
        df_normal_downsampled_data = df_node.sample(frac=frac, random_state=42)

        df_balance = pd.concat((df_balance, df_normal_downsampled_data), ignore_index=True)

    df_balance = pd.concat((df_itc, df_balance), ignore_index=True)
    df_balance.to_csv('tmp.csv', index=False)

    return df_balance

