import os
import random
import functools
from collections import OrderedDict

import numpy as np
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from enum import Enum
import json

def apply_grad(model, grad):
    '''
    assign gradient to model(nn.Module) instance. return the norm of gradient
    '''
    grad_norm = 0
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
        grad_norm += torch.sum(g ** 2)
    grad_norm = grad_norm ** (1 / 2)
    return grad_norm.item()


def mix_grad(grad_list, weight_list):
    '''
    calc weighted average of gradient
    '''
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
        mixed_grad.append(torch.sum(g_list, dim=0))
    return mixed_grad


def set_seed(seed):
    # 保证每次运行结果一样
    # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。如果设置初始化，则每次初始化都是固定的。
    # for reproducibility.
    # note that pytorch is not completely reproducible
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = False  # 可以GPU提速CNN
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.initial_seed()  # dataloader multi processing,设置随机数种子，每次生成的随机数都一样
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def set_gpu(x):
    x = [str(e) for e in x]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(x)
    print('using gpu:', ','.join(x))


def check_dir(args):
    # save path
    path = os.path.join(args.result_path, args.alg)
    if not os.path.exists(path):
        os.makedirs(path)
    return None


# https://github.com/sehkmg/tsvprint/blob/master/utils.py
def dict2tsv(res, file_name):
    if not os.path.exists(file_name):
        with open(file_name, 'a') as f:
            f.write('\t'.join(list(res.keys())))
            f.write('\n')

    with open(file_name, 'a') as f:
        f.write('\t'.join([str(r) for r in list(res.values())]))
        f.write('\n')


class BestTracker:
    '''Decorator for train function.
       Get ordered dict result (res),
       track best dice coef (self.best_dice & best epoch (self.best_epoch) and
       append them to ordered dict result (res).
       Also, save the best result to file (best.txt).
       Return ordered dict result (res).'''

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.best_epoch = 0
        self.best_test_dice = 0

    def __call__(self, *args, **kwargs):
        res = self.func(*args, **kwargs)

        if res['test_dice'] > self.best_test_dice:
            self.best_epoch = res['epoch']

            self.best_test_dice = res['test_dice']
            is_best = True
        else:
            is_best = False

        res['best_epoch'] = self.best_epoch

        res['best_test_dice'] = self.best_test_dice

        return res, is_best

        def get_confusion_matrix_elements(groundtruth_list, predicted_list):
            """returns confusion matrix elements i.e TN, FP, FN, TP as floats

            """
            predicted_list = np.round(predicted_list).astype(int)
            groundtruth_list = np.round(groundtruth_list).astype(int)
            groundtruth_list = groundtruth_list.reshape(-1)
            predicted_list = predicted_list.reshape(-1)
            tn, fp, fn, tp = confusion_matrix(groundtruth_list, predicted_list).ravel()
            tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)

            return tn, fp, fn, tp

        def get_mcc(groundtruth_list, predicted_list):
            """Return mcc covering edge cases"""

            tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)
            mcc = ((tp * tn) - (fp * fn)) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
            return mcc

        def get_precision(groundtruth_list, predicted_list):

            tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)

            total = tp + fp + fn + tn
            accuracy = (tp + tn) / total

            return accuracy

        ################
        def log_images(x, y_pred, y_true=None, channel=1):
            images = []
            x_np = x[:, channel].cpu().numpy()
            y_true_np = y_true[:, 0].cpu().numpy()
            y_pred_np = y_pred[:, 0].cpu().numpy()
            for i in range(x_np.shape[0]):
                image = gray2rgb(np.squeeze(x_np[i]))
                image = outline(image, y_pred_np[i], color=[255, 0, 0])
                image = outline(image, y_true_np[i], color=[0, 255, 0])
                images.append(image)
            return images

        def gray2rgb(image):
            w, h = image.shape
            image += np.abs(np.min(image))
            image_max = np.abs(np.max(image))
            if image_max > 0:
                image /= image_max
            ret = np.empty((w, h, 3), dtype=np.uint8)
            ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255
            return ret

        def outline(image, mask, color):
            mask = np.round(mask)
            yy, xx = np.nonzero(mask)
            for y, x in zip(yy, xx):
                if 0.0 < np.mean(mask[max(0, y - 1): y + 2, max(0, x - 1): x + 2]) < 1.0:
                    image[max(0, y): y + 1, max(0, x): x + 1] = color
            return image


def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if opt.lr_mode == 'step':
        lr = opt.lr * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr * (1 - epoch / opt.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# 传入要创建的文件夹路径即可
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

import shutil
def remove_folder(path):
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)
    else:
         # throw your exception to handle this special scenario
         # raise RuntimeError ("your exception")
        pass


def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def save_args_to_file(args, filename):
    # 将 argparse 命名空间转换为字典
    args_dict = vars(args)

    # 将参数字典保存为 JSON 文件
    with open(filename, 'w') as file:
        json.dump(args_dict, file, indent=2)