'''
所有中心，ffpe, 4X-normal, train 和test patch减少50% 
'''

'''

'''
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')
import os
import pandas as pd
from common import utils
import numpy as np


# 输入一个class的df
def shiye_num(df, index_begin=0):
    screen_ls = []
    for _,row in df.iterrows():
        path = row[-1]
        path_ls = path.split('/')
        screen_ls.append(path_ls[2+index_begin] +'/' + path_ls[3+index_begin] + '/' +  path_ls[4+index_begin])
    
    
    return len(np.unique(screen_ls))

def tongji(res, dframe):
    
    for i in res:
        df_res = dframe[dframe['Resolution'] == i]
        class_ls = np.unique(list(df_res['Class']))

        for class_name in class_ls:
            df_class = df_res[df_res['Class'] == class_name]
            wsi_list = np.unique(list(df_class['WSI']))
            print(f'resilution_{i} class_{class_name} wsi num: ', len(wsi_list))
            node_list = np.unique(list(df_class['Node']))
            print(f'resilution_{i} class_{class_name} nodes num: ', len(node_list))
            print(f'resilution_{i} class_{class_name} screen num: ', shiye_num(df_class, index_begin=2))
            print(f'resilution_{i} class_{class_name} patches num: ', len(df_class))

            print(len(wsi_list), len(node_list), shiye_num(df_class), len(df_class))



def multi_center_balance():

    # zhongshan数据太特殊，单独平衡
    center_ls = ['nandafuyi','zhongshaneryuan', 'zhongshansanyuan']
    dis_ls = ['ffpe', 'bf']
    res_ls = ['4x', '10x']

    dataset_ls = ['train', 'val', 'test']

    for center in center_ls:
        for dis in dis_ls:
            for res in res_ls:
                store_dir =f'/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/{center}/{dis}/split_by_node/{res}/balance/' 
                utils.mkdir(store_dir)
                for dataset in dataset_ls:
                    frac = 1
                    if  res == '4x':
                        if dataset == 'train':
                            frac = 0.5
                        if dis == 'ffpe' and dataset == 'test':
                            frac = 0.3
                        if dis == 'bf' and dataset == 'test':
                            frac = 0.5


                    dframe = pd.read_csv(f'/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/{center}/{dis}/split_by_node/{res}/{dataset}_{res}.csv')
                    store_path =os.path.join(store_dir,f'{dataset}_{res}.csv')
                    

                    df_normal = dframe[dframe['Class'] == f'{dis}_normal']
                    df_micro = dframe[dframe['Class'] == f'{dis}_micro']

                    all_nodes = df_normal['Node'].unique()

                    df_balance= pd.DataFrame()
                    for node in all_nodes:
                        df_node = df_normal[df_normal['Node'] == node]
                        # 获取数据框的行数
                        num_rows = df_node.shape[0]
                        # 随机选择一部分行，可以根据需要调整 frac 的值
                        frac_for_downsampling = frac # 0.1选择原始行数的10%, 0.2 20%
                        downsampled_data = df_node.sample(frac=frac_for_downsampling, random_state=42)

                        df_balance = pd.concat((df_balance, downsampled_data), ignore_index=True)

                    df_balance = pd.concat((df_balance, df_micro), ignore_index=True)


                    df_balance.to_csv(store_path, index=False)

                    print(f'{dataset}++++++++++++++')
                    tongji([res], df_balance)


def balance_for_zhongshan():

    # zhongshan数据太特殊，单独平衡
    center_ls = ['zhongshan']
    dis_ls = ['ffpe', 'bf']
    res_ls = ['4x', '10x']

    dataset_ls = ['train', 'val', 'test']

    for center in center_ls:
        for dis in dis_ls:
            for res in res_ls:
                

                store_dir =f'/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/{center}/{dis}/split_by_node/{res}/balance/' 
                utils.mkdir(store_dir)
                for dataset in dataset_ls:
                    frac = 1
                    if dis == 'bf' and res == '4x' and dataset == 'train':
                        frac = 0.5
                    
                    if dis == 'ffpe' and res == '10x' and dataset == 'train':
                        frac = 0.4

                    if dis == 'ffpe' and res == '10x' and dataset == 'val':
                        frac = 0.15
                    
                    if dis == 'ffpe' and res == '10x' and dataset == 'test':
                        frac = 0.1
                    
                    if dis == 'ffpe' and res == '4x' and dataset == 'train':
                        frac = 0.1
                    
                    if dis == 'ffpe' and res == '4x' and dataset == 'val':
                        frac = 0.1
                    
                    if dis == 'ffpe' and res == '4x' and dataset == 'test':
                        frac = 0.05
                    
                        
                    dframe = pd.read_csv(f'/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/{center}/{dis}/split_by_node/{res}/{dataset}_{res}.csv')
                    store_path =os.path.join(store_dir,f'{dataset}_{res}.csv')
                    

                    df_normal = dframe[dframe['Class'] == f'{dis}_normal']
                    df_micro = dframe[dframe['Class'] == f'{dis}_micro']

                    all_nodes = df_normal['Node'].unique()

                    df_balance= pd.DataFrame()
                    for node in all_nodes:
                        df_node = df_normal[df_normal['Node'] == node]
                        # 获取数据框的行数
                        num_rows = df_node.shape[0]
                        # 随机选择一部分行，可以根据需要调整 frac 的值
                        frac_for_downsampling = frac  # 0.1选择原始行数的10%, 0.2 20%
                        downsampled_data = df_node.sample(frac=frac_for_downsampling, random_state=42)

                        df_balance = pd.concat((df_balance, downsampled_data), ignore_index=True)

                    df_balance = pd.concat((df_balance, df_micro), ignore_index=True)


                    df_balance.to_csv(store_path, index=False)

                    print(f'{dataset}++++++++++++++')
                    tongji([res], df_balance)

balance_for_zhongshan()
# multi_center_balance()