import pandas as pd
import numpy as np

# df_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/source_data/source_classes.csv'
df_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/source_data/tissue/tissue.csv'
# df_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/source_data/tissue_and_source.csv'
dframe = pd.read_csv(df_path)
res = ['4x', '10x']

# 输入一个class的df
def shiye_num(df):
    screen_ls = []
    for path in df['Image_path']:
        path_ls = path.split('/')
        screen_ls.append(path_ls[3] + '/' +  path_ls[4])
    
    
    return len(np.unique(screen_ls))

for i in res:
    df_res = dframe[dframe['Resolution'] == i]
    class_ls = np.unique(list(df_res['Class']))

    for class_name in class_ls:
        df_class = df_res[df_res['Class'] == class_name]
        wsi_list = np.unique(list(df_class['WSI']))
        print(f'resilution_{i} class_{class_name}')
        # print(f'resilution_{i} class_{class_name} wsi num: ', len(wsi_list))
        # print(f'resilution_{i} class_{class_name} screen num: ', shiye_num(df_class))
        # print(f'resilution_{i} class_{class_name} patches num: ', len(df_class))
        print(len(wsi_list))
        print(shiye_num(df_class))
        print(len(df_class))
        

