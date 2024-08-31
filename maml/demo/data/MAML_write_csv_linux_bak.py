import os
import csv

# train_class = all_class[:59]
# val_class = all_class[60:79]
# test_class = all_class[80:]


def write_train_csv():
    root = 'D:/workplace/python/data/meta-oct/seg/faker_CNV'
    source_list = []
    for root, dirs, files in os.walk(root, topdown=True):
        source_list = dirs
        break

    with open('train_data.csv', 'w', newline='') as csvfile:
        fields = ['ID', 'Image_path', 'Label_path']
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(fields)
        for idx, each_set in enumerate(source_list[:80]):
            for image in os.listdir(root + '/' + each_set + '/' + 'y') and os.listdir(
                    root + '/' + each_set + '/' + 'x'):
            # for image in os.listdir(root + '/' + each_set + '/' + 'y'):
                label_path = root + '/' + each_set + '/' + 'y' + '/' + image.replace('.jpg', '_label.png')
                image_path = root + '/' + each_set + '/' + 'x' + '/' + image
                csvwriter.writerow([idx, image_path, label_path])

    with open('val_data.csv', 'w', newline='') as csvfile:
        fields = ['ID', 'Image_path', 'Label_path']
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(fields)
        for idx, each_set in enumerate(source_list[80:]):
            for image in os.listdir(root + '/' + each_set + '/' + 'y') and os.listdir(
                    root + '/' + each_set + '/' + 'x'):
                label_path = root + '/' + each_set + '/' + 'y' + '/' + image.replace('.jpg', '_label.png')
                image_path = root + '/' + each_set + '/' + 'x' + '/' + image
                csvwriter.writerow([idx+80, image_path, label_path])


def write_target_csv():
    root = 'D:/workplace/python/data/meta-oct/seg'
    target_set = 'CNV'
    with open('test_data.csv', 'w', newline='') as csvfile:
        fields = ['ID', 'Image_path', 'Label_path']
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(fields)
        for image in os.listdir(root + '/' + target_set + '/' + 'y') and os.listdir(
                root + '/' + target_set + '/' + 'x'):
            label_path = root + '/' + target_set + '/' + "y" + '/' + image.replace('.jpg', '.png')
            image_path = root + '/' + target_set + '/' + "x" + '/' + image
            csvwriter.writerow([target_set, image_path, label_path])


if __name__ == '__main__':
    write_train_csv()

