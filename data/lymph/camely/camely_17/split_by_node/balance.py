'''
当前数据mircro 和 normal不均衡,进行均衡化
'''

'''

'''
import pandas as pd
import os
import numpy as np


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
            print(f'resilution_{i} class_{class_name} screen num: ', '/')
            print(f'resilution_{i} class_{class_name} patches num: ', len(df_class))

            print(len(wsi_list), len(node_list), '/', len(df_class))


class_ls = ['ffpe_normal', 'ffpe_micro', 'ffpe_macro', 'ffpe_itc']

frac_d_4x = {'ffpe_itc':{'train':1, 'val':1, 'test':1},
          'ffpe_macro':{'train':1, 'val':1, 'test':1},
          'ffpe_micro':{'train':1, 'val':1, 'test':1},
          'ffpe_normal':{'train':0.08, 'val':0.05, 'test':0.2}, }

frac_d_10x = {'ffpe_itc':{'train':1, 'val':1, 'test':1},
          'ffpe_macro':{'train':1, 'val':1, 'test':1},
          'ffpe_micro':{'train':1, 'val':1, 'test':1},
          'ffpe_normal':{'train':0.2, 'val':0.1, 'test':0.3}, }

frac_d = {'4x': frac_d_4x, '10x': frac_d_10x}

res = '4x'
csv_file_types = [f'train_{res}.csv', f'val_{res}.csv', f'test_{res}.csv']

path_pre = f'/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/camely/camely_17/splite_by_node/{res}'
for csv_file in csv_file_types:
    csv_file_path = os.path.join(path_pre, csv_file)
    store_path =  os.path.join(path_pre, 'balance', csv_file)
    dframe = pd.read_csv(csv_file_path)
    df_normal = dframe[dframe['Class'] == 'ffpe_normal']
    df_micro = dframe[dframe['Class'] == 'ffpe_micro']
    df_macro = dframe[dframe['Class'] == 'ffpe_macro']
    df_itc = dframe[dframe['Class'] == 'ffpe_itc']

    
    
    df_balance= pd.DataFrame()
    nor_all_nodes = df_normal['Node'].unique()
    for node in nor_all_nodes:
        df_node = df_normal[df_normal['Node'] == node]
        df_normal_downsampled_data = df_node.sample(frac=frac_d[res]['ffpe_normal'][csv_file.split('_')[0]], random_state=42)

        df_balance = pd.concat((df_balance, df_normal_downsampled_data), ignore_index=True)

    df_balance = pd.concat((df_itc, df_macro, df_micro, df_balance), ignore_index=True)
    df_balance.to_csv(store_path, index=False)

    print(csv_file, '+++++++++++')
    tongji([res], df_balance)

