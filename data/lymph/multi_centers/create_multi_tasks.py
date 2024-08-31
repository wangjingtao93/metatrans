import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')
import os
import pandas as pd
from common import utils
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import GroupKFold



# 用balance 之后的
relative_path= '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication'

def creat_multi_tasks(store_path, df_train, df_val):
    # store_path = os.path.join(relative_path, 'data/lymph/breast/FFPE/split_by_node/10x/balance/multi_tasks/')
    # os.mkdir(store_path)
    # df_train = pd.read_csv(os.path.join(relative_path,'data/lymph/breast/FFPE/split_by_node/10x/balance/train_10x.csv'))
    # df_val = pd.read_csv(os.path.join(relative_path,'data/lymph/breast/FFPE/split_by_node/10x/balance/val_10x.csv'))

    new_df = pd.DataFrame()
    new_df = pd.concat([df_train, df_val],ignore_index=True)

    groups = new_df['Node']  # 作为GroupKFold的groups参数

    k_fold_num = 5
    kf = GroupKFold(n_splits=k_fold_num)
    k_fold_count = 0

    for fold, (train_index, val_index) in enumerate(kf.split(X=new_df, y=new_df['Class'], groups=groups), 1):
        print('\n{} of kfold {}'.format(k_fold_count,kf.n_splits))
        
        S_df = new_df.iloc[train_index]
        Q_df = new_df.iloc[val_index]

        S_df.to_csv(os.path.join(store_path, f't_{k_fold_count}_s.csv'), index=False)
        Q_df.to_csv(os.path.join(store_path, f't_{k_fold_count}_q.csv'),index=False)

        k_fold_count += 1

        # end-------------------- use code/process to splide
    

# 测试 S和Q是否有重复的node
def test_s_q():
    df_s = pd.read_csv(os.path.join(relative_path,'data/lymph/multi_centers/nandafuyi/ffpe/split_by_node/4x/multi_tasks/t_1_s.csv'))
    df_q = pd.read_csv(os.path.join(relative_path,'data/lymph/multi_centers/nandafuyi/ffpe/split_by_node/4x/multi_tasks/t_1_q.csv'))

    class_name = df_s['Class'].unique()
    for c in class_name:
        df_s_c = df_s[df_s['Class'] == c]
        node_ls_s = df_s_c['Node'].unique()

        df_q_c = df_q[df_q['Class'] == c]
        node_ls_q = df_q_c['Node'].unique()

        for node_s in node_ls_s:
            if node_s in node_ls_q:
                print('nonono', node_s)

# 测试两折的Q是否有重复的node
 # 测试 S和Q是否有重复的node
def test_q():
    df_q_1 = pd.read_csv(os.path.join(relative_path,'data/lymph/multi_centers/nandafuyi/ffpe/split_by_node/4x/multi_tasks/t_1_s.csv'))
    df_q_2 = pd.read_csv(os.path.join(relative_path,'data/lymph/multi_centers/nandafuyi/ffpe/split_by_node/4x/multi_tasks/t_1_q.csv'))

    class_name = df_q_1['Class'].unique()
    for c in class_name:
        df_s_c = df_q_1[df_q_1['Class'] == c]
        node_ls_q1 = df_s_c['Node'].unique()

        df_q_c = df_q_2[df_q_2['Class'] == c]
        node_ls_q2 = df_q_c['Node'].unique()

        for node_s in node_ls_q1:
            if node_s in node_ls_q2:
                print('nonono', node_s)               

# creat_multi_tasks()           
# test_s_q()
# test_q()


def multi_center_create():
    center_ls = ['nandafuyi', 'zhongshan', 'zhongshaneryuan', 'zhongshansanyuan']
    dis_ls = ['ffpe', 'bf']
    res_ls = ['4x','10x']

    for center in center_ls:
        for dis in dis_ls:
            for res in res_ls:
                # 使用均衡后的数据
                path_prefix =f'/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/{center}/{dis}/split_by_node/{res}/balance/'

                # 所有中心，ffpe, 4x-normal, train 和test patch都经过了平衡
                # 使用平衡后的
                
                df_train = pd.read_csv(os.path.join(path_prefix,f'train_{res}.csv'))
                df_val = pd.read_csv(os.path.join(path_prefix,f'val_{res}.csv'))

                df_train.loc[:, 'Center'] = center 
                df_val.loc[:, 'Center'] = center 

                store_path = os.path.join(path_prefix, 'multi_tasks')
                utils.mkdir(store_path)

                creat_multi_tasks(store_path, df_train, df_val)
                
multi_center_create()

test_s_q()