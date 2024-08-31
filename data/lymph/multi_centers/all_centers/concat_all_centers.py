import pandas as pd
import os


center_ls = ['zhongshan','nandafuyi', 'zhongshaneryuan', 'zhongshansanyuan']
dis_ls = ['ffpe','bf']
res_ls = ['4x', '10x']

# concat 所有中心的tasks

def concat_multi_tasks():
    
    for dis in dis_ls:
        for res in res_ls:
            store_path = f'/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/all_centers/multi_tasks/{dis}_{res}/'

            for i in range(5):
                dframe_s_c = pd.DataFrame()
                dframe_q_c = pd.DataFrame()
                for center in center_ls:
                    path_prefix =f'/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/{center}/{dis}/split_by_node/{res}/balance/multi_tasks'
                    # 五折
                    dframe_s = pd.read_csv(os.path.join(path_prefix,f't_{i}_s.csv' ))
                    dframe_q = pd.read_csv(os.path.join(path_prefix,f't_{i}_q.csv' ))

                    dframe_s_c = pd.concat([dframe_s_c, dframe_s],ignore_index=True)
                    dframe_q_c = pd.concat([dframe_q_c, dframe_q],ignore_index=True)

                
                dframe_s_c.to_csv(os.path.join(store_path, f't_{i}_s.csv'), index=False)
                dframe_q_c.to_csv(os.path.join(store_path, f't_{i}_q.csv'), index=False)

def concat_test():
    for dis in dis_ls:
        for res in res_ls:
            store_path = f'/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/all_centers/multi_tasks/{dis}_{res}/'
            dframe_test_c = pd.DataFrame()
            for center in center_ls:
                path_prefix =f'/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/{center}/{dis}/split_by_node/{res}/balance'
                dframe_test = pd.read_csv(os.path.join(path_prefix,f'test_{res}.csv'))
                
                dframe_test_c = pd.concat([dframe_test_c, dframe_test], ignore_index=True)
            
            dframe_test_c.to_csv(os.path.join(store_path, 'test.csv'), index=False)


if __name__ == '__main__':
    concat_multi_tasks()
    concat_test()