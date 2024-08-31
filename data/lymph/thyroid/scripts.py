import pandas as pd
import glob
import os
import csv
def func_1():
    dframe = pd.read_csv('all.csv')

    df_4x = dframe[dframe['Resolution'] == '4x']

    df_4x.to_csv('4x.csv', index=False)

    df_10x = dframe[dframe['Resolution'] == '10x']
    df_10x.to_csv('10x.csv', index=False)

# crc thriod, for heatmap
def gen_csv():
    img_dir = '/data1/wangjingtao/workplace/python/data/classification/lymph/throid_crc_forheatmap/throid'
    
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
    
            class_name = value_ls[2].split('-')[0] + '_micro'

            center = 'zhongshan'
            wsi = value_ls[-1].split('(')[0]         
            path = img
            datawrite.writerow([i+1, wsi, resolution, center, class_name, path]) 

gen_csv()
