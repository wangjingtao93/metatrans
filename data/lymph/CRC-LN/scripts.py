import pandas as pd
import os
import csv
import glob




def balance():
    dframe = pd.read_csv('all.csv')
    res = ['4x', '10x']
    for i in res:
        df_res = dframe[dframe['Resolution']==i]

        
        df_normal = df_res[df_res['Class'] == 'ffpe_normal']
        df_micro = df_res[df_res['Class'] == f'ffpe_micro']

        if i == '4x':
            frac = 0.1
        if i == '10x':
            frac = 0.05

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

        df_balance.to_csv(f'micro_{i}_balance.csv', index=False)

# crc thriod, for heatmap
def gen_csv():
    img_dir = '/data1/wangjingtao/workplace/python/data/classification/lymph/throid_crc_forheatmap/crc'
    
    relative_path = '/data1/wangjingtao/workplace/python/data/classification/lymph/'
    

    with open('tmp.csv', 'w', newline='') as datacsvfile:
        fields = ['ID', 'WSI', 'Resolution', 'Center', 'Class', 'Image_path']
        datawrite = csv.writer(datacsvfile, delimiter=',')
        datawrite.writerow(fields)
        
        img_ls = glob.glob(os.path.join(img_dir, "**", "*.jpg"),
                        recursive=True)
        
        
        img_ls = img_ls


        for i, img in enumerate(img_ls):
            img = img.replace(relative_path, '')
            value_ls = img.split('/')
            
            resolution =  value_ls[2].split('-')[-1].lower()
    
            class_name =  value_ls[2].split('-')[0] + '_micro'

            center = 'zhongshan'

            wsi = value_ls[-1].split('(')[0]         
            path = img
            datawrite.writerow([i+1, wsi, resolution,center, class_name, path]) 

gen_csv()

# balance()