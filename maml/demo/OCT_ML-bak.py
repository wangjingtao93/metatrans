import torch, os
import numpy as np
from MiniImagenet import MiniImagenet
import scipy.stats
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse
from dataloader_OCTimage import OCTImage

from learner import Learner
from torch import optim
from meta import Meta
from copy import deepcopy
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter


def ml_train(x_spt, y_spt, x_qry, y_qry):
    querysz = x_qry.size(0)
    corrects = [0 for _ in range(args.update_step_test + 1)]

    # in order to not ruin the state of running_mean/variance and bn_weight/bias
    # we finetunning on the copied model instead of self.net
    # 保证多个任务的梯度更新之间不会互相影像，即每个任务都从最开始更新梯度
    net = deepcopy(maml.net)

    # 1. run the i-th task and compute loss for k=0
    logits = net(x_spt)
    loss = F.cross_entropy(logits, y_spt)
    grad = torch.autograd.grad(loss, net.parameters())
    fast_weights = list(map(lambda p: p[1] - args.update_lr * p[0], zip(grad, net.parameters())))

    # this is the loss and accuracy before first update
    with torch.no_grad():
        # [setsz, nway]
        logits_q = net(x_qry, net.parameters(), bn_training=True)
        # [setsz]
        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        # scalar
        correct = torch.eq(pred_q, y_qry).sum().item()
        corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
    with torch.no_grad():
        # [setsz, nway]
        logits_q = net(x_qry, fast_weights, bn_training=True)
        # [setsz]
        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        # scalar
        correct = torch.eq(pred_q, y_qry).sum().item()
        corrects[1] = corrects[1] + correct

    # 2到k-1次梯度更新后的loss和acc
    for k in range(1, args.update_step_test):
        # 1. run the i-th task and compute loss for k=1~K-1
        logits = net(x_spt, fast_weights, bn_training=True)
        loss = F.cross_entropy(logits, y_spt)
        # 2. compute grad on theta_pi
        grad = torch.autograd.grad(loss, fast_weights)
        # 3. theta_pi = theta_pi - train_lr * grad
        fast_weights = list(map(lambda p: p[1] - args.update_lr * p[0], zip(grad, fast_weights)))
        logits_q = net(x_qry, fast_weights, bn_training=True)
        # loss_q will be overwritten and just keep the loss_q on last update step.
        loss_q = F.cross_entropy(logits_q, y_qry)

        with torch.no_grad():
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
            corrects[k + 1] = corrects[k + 1] + correct
            writer.add_scalar('acc', correct / querysz, k, walltime=None)

    del net

    accs = np.array(corrects) / querysz

    return accs


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=4)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=150)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=40)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10000)
    args = argparser.parse_args()

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]
    device = torch.device('cuda')
    # ori_net = Learner(config, args.imgc, args.imgsz)

    # ml_optim = optim.Adam(ori_net.parameters(), lr=args.meta_lr)
    # ori_net.to(device)
    maml = Meta(args, config)
    print(maml)
    # maml.load_state_dict(torch.load('./model_ckpt/mini-imagenet-2way-5shots-15query/model_5_2000.ckpt'))
    # maml.load_state_dict(torch.load('./model_ckpt/model_5_0.ckpt'))
    maml.to(device)

    # mini_test = MiniImagenet('D:/workplace/python/data/mini-imagenet/', mode='test', n_way=4, k_shot=5,
    #                          k_query=15,
    #                          batchsz=100, resize=args.imgsz)
    mini_test = OCTImage('D:/workplace/python/data/meta-oct/4-ways-1000-fromTraining/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                         k_query=args.k_qry,
                         batchsz=1, resize=args.imgsz)
    db_test = DataLoader(mini_test, 5, shuffle=True, num_workers=0, pin_memory=True)
    accs_all_tasks = []

    writer = SummaryWriter('./tensorboard_log')

    for x_spt, y_spt, x_qry, y_qry in db_test:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        task_acc = ml_train(x_spt, y_spt, x_qry, y_qry)
        # task_acc = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
        # accs_all_tasks.append(task_acc)
    # accs = np.array(accs_all_tasks).mean(axis=0).astype(np.float16)
    # print('Test acc:', accs)
