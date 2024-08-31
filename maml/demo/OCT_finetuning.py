import torch, os
import numpy as np
from MiniImagenet import MiniImagenet
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse
from dataloader_OCTimage import OCTImage

from meta import Meta
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=4)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=105)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=35)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=500)
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
    maml = Meta(args, config)
    print(maml)
    # maml.load_state_dict(torch.load('./model_ckpt/mini-imagenet-2way-5shots-15query/model_5_2000.ckpt'))
    # maml.load_state_dict(torch.load('./model_ckpt/model_5_0.ckpt'))
    # maml.load_state_dict(torch.load('./model_ckpt/mini-imagenet-2way-5shots-15query/model_5_2000.ckpt'))
    maml.to(device)

    # mini_test = MiniImagenet('D:/workplace/python/data/mini-imagenet/', mode='test', n_way=4, k_shot=5,
    #                          k_query=15,
    #                          batchsz=100, resize=args.imgsz)
    # batchsz构造的任务的个数
    # mini_test = OCTImage('D:/workplace/python/data/meta-oct/classic/4-ways/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
    #                     k_query=args.k_qry,
    #                      batchsz=3, resize=args.imgsz)
    mini_test = OCTImage('data/train_data.csv', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                         batchsz=3, resize=args.imgsz)
    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=0, pin_memory=True)
    accs_all_tasks = []

    for x_spt, y_spt, x_qry, y_qry in db_test:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        task_accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
        accs_all_tasks.append(task_accs)


    accs = np.array(accs_all_tasks).mean(axis=0).astype(np.float16)
    print('Test acc:', accs)

    writer = SummaryWriter('tensorboard_log')
    for i in range(len(accs)):
        writer.add_scalar('Test acc', accs[i].item(), i, walltime=None)
