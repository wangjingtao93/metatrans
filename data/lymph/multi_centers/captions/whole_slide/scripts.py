import pandas as pd
import csv
import glob
import os
img_dir = '/data1/wangjingtao/workplace/python/data/classification/lymph/multi_centers/captions/whole_slide/10x'
relative_path = '/data1/wangjingtao/workplace/python/data/classification/lymph/'
class_name_prefix = 'ffpe_'
with open('whole_slide.csv', 'w', newline='') as datacsvfile:
    fields = ['ID','Center','Resolution', 'Class', 'Image_path']
    datawrite = csv.writer(datacsvfile, delimiter=',')
    datawrite.writerow(fields)
    
    img_ls = glob.glob(os.path.join(img_dir, "**", "*.jpg"),
                    recursive=True)
    img_ls_png = glob.glob(os.path.join(img_dir, "**", "*.png"),
            recursive=True)
    
    img_ls = img_ls + img_ls_png

    for i, img in enumerate(img_ls):
            img = img.replace(relative_path, '')
            resolution =  '10x'
            class_name =  'ffpe_micro'
            center = 'someone'
            

            path = img.replace(relative_path, '')
            datawrite.writerow([i+1, center, resolution, class_name, path])
