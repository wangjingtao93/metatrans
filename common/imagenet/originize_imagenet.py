import pandas as pd
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split



def originze_imagenet():
    root = '/data1/wangjingtao/workplace/python/data/meta-oct/classic/ILSVRC2012'
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )
    test_query_fileroots_alltask, test_support_fileroots_all_task = [], []
    final_test_alltask = []#用于最终测试
    test_data_ls = []

    test_support_fileroots_all_task.append(train_dataset)
    test_query_fileroots_alltask.append(val_dataset)
    final_test_alltask.append(val_dataset)



    test_data_ls = [test_support_fileroots_all_task, test_query_fileroots_alltask, final_test_alltask]

    train_data_ls = []

    return  test_data_ls, train_data_ls
