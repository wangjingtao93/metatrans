import pandas as pd

# 选择测试集里的镜下

alljingxia = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/captions/feijingxia/all_captions.csv'




dframe = pd.read_csv(alljingxia)

res_l= ['4x', '10x']
class_name_l = ['ffpe']


df_new_test = pd.DataFrame()
for res in res_l:
    df_all_res = dframe[dframe['Resolution']==res]
    wsi_ls = []
    for class_name in class_name_l:
    
        test_file =   f'/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/all_centers/multi_tasks/{class_name}_{res}/test.csv'

        df_test_class_res = pd.read_csv(test_file)

        wsi_ls += df_test_class_res['WSI'].tolist()
    
        wsi_ls_tmp = df_all_res['WSI'].tolist()


    df_new_test = pd.concat([df_new_test, df_all_res[df_all_res['WSI'].isin(wsi_ls)]],ignore_index=True)


df_new_test.to_csv('tmp.csv', index=False)







