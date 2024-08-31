import os
import csv
import pandas as pd


# train_class = all_class[:59]
# val_class = all_class[60:79]
# test_class = all_class[80:]


def write_csv():
    root = 'D:/workplace/python/data/meta-oct/classic/st/Triton'
    source_list = []
    for root, dirs, files in os.walk(root, topdown=True):
        source_list = dirs
        break
    print("疾病种类=", source_list)
    # source_list.sort(key=lambda element: int(element.split('_')[-1]))
    with open('all_data.csv', 'w', newline='') as csvfile:
        fields = ['ID', 'Image_path', 'Label', 'Eye']
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(fields)

        for idx, each_set in enumerate(source_list):
            scan_number = 0
            zy_list = os.listdir(root + '/' + each_set)

            print("疾病  ", each_set, "眼睛数量 ", len(zy_list))
            for eye_name in zy_list:
                image_list_tmp = os.listdir(root + '/' + each_set + '/' + eye_name)
                image_list = []
                for i, image in enumerate(image_list_tmp):
                    # if not os.path.isfile(root + '/' + each_set + '/' + eye_name + '/' + image):
                    #     print('+++++++++++++++')
                    #     print(root + '/' + each_set + '/' + eye_name)
                    #
                    #     print('--------------')
                    if os.path.isfile(root + '/' + each_set + '/' + eye_name + '/' + image):
                        image_list.append(image)
                image_list.sort(key=lambda element: int(element.replace('.jpg', '').split('_')[-1]))
                scan_number += len(image_list)
                for image in image_list:
                    image_path = 'Triton/' + each_set + '/' + eye_name + '/' + image
                    label = each_set
                    csvwriter.writerow([idx, image_path, label, eye_name])

            print("疾病  ", each_set, "scan数量 ", scan_number)


def get_train_test_csv():
    csv_dir = "D:/workplace/python/metaLearning/MAML-Pytorch-master-dragen/data/win/all_data.csv"
    dataframe = pd.read_csv(csv_dir)

    # img_file = dataframe["Image_path"]
    # mask_file = dataframe["Label"]
    dframe_PIC = dataframe[dataframe["Label"] == 'PIC']
    dframe_normal = dataframe[dataframe["Label"] == 'normal']
    dframe_RP = dataframe[dataframe["Label"] == 'RP']
    dframe_PVRL = dataframe[dataframe["Label"] == 'PVRL']

    dframe_train = dataframe[dataframe["Label"] != 'PIC']
    dframe_train = dframe_train[dframe_train["Label"] != 'normal']
    dframe_train = dframe_train[dframe_train["Label"] != 'RP']
    dframe_train = dframe_train[dframe_train["Label"] != 'PVRL']

    dframe_normal.to_csv('normal.csv', index=0)
    dframe_PIC.to_csv('PIC.csv', index=0)  # 不保留行索引
    dframe_RP.to_csv('RP.csv', index=0)
    dframe_train.to_csv('train_data.csv', index=0)

    train_classes = set(dframe_train['Label'])


    dframe_test_RP_Normal = pd.concat([dframe_RP, dframe_normal])
    dframe_test_PIC_Normal = pd.concat([dframe_PIC, dframe_normal])
    dframe_test_PVRL_Normal = pd.concat([dframe_PVRL, dframe_normal])

    dframe_test_RP_Normal.to_csv('test_RP_Normal.csv', index=0)
    dframe_test_PIC_Normal.to_csv('test_PIC_Normal.csv', index=0)
    dframe_test_PVRL_Normal.to_csv('test_PVRL_Normal.csv', index=0)


    print('nihao')

if __name__ == '__main__':
    # write_csv()
    get_train_test_csv()
