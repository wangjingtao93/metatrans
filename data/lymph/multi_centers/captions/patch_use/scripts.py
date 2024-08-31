# 将数据按照分辨率和ffpe分开

import pandas as pd

import os
dframe = pd.read_csv('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/captions/patch_use/jingxia_multi_centers.csv')

res = ['4x', '10x']

class_ls = ['bf', 'ffpe']

def split_res():
    for i in res:
        df_res = dframe[dframe['Resolution']==i]

        df_res.to_csv(f'{i}.csv', index=False)

def balance():
    for i in res:
        df_res = dframe[dframe['Resolution']==i]

        for dis in class_ls:
            df_normal = df_res[df_res['Class'] == f'{dis}_normal']
            df_micro = df_res[df_res['Class'] == f'{dis}_micro']

            if dis == 'bf' and i == '4x':
                frac = 0.25
            elif  dis == 'bf' and i == '10x':
                frac = 0.4

            elif  dis == 'ffpe' and i == '4x':   
                frac = 0.25

            elif  dis == 'ffpe' and i == '4x':
                frac = 0.35 

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

            df_balance.to_csv(f'{dis}_{i}_balance.csv', index=False)

            # 统计
            df_balance_normal =df_balance[df_balance['Class'] == f'{dis}_normal']
            print(f'{i}_{dis}_bal_normal_patches: ', len(df_balance_normal))

    
# balance()
split_res()

