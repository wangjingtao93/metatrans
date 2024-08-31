import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')

from timm.models import create_model
import argparse
import timm_register_models
import torch
import torch.nn as nn
from model.conformer import Conformer
from model.resnet import ResNet18
import torchvision

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='mt_small_model_lymph', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
    parser.add_argument('--num_classes', type=int, default=2)# Total number of fc out_class
    parser.add_argument('--depth', type=int, default=12)# Total number of fc out_class
    return parser



parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
device = torch.device('cuda')
model = create_model(
    args.model,
    pretrained=False,
    num_classes=args.num_classes,
    # depth=args.depth
).to(device)

# model = ResNet18(num_classes=4).to(device)
# model = torchvision.models.resnet18(pretrained=False)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 4)

# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)
# model = nn.Sequential(
#     nn.Conv2d(1, 64, 3),
#     nn.BatchNorm2d(64, momentum=1, affine=True),
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(2, 2),
#     nn.Conv2d(64, 64, 3),
#     nn.BatchNorm2d(64, momentum=1, affine=True),
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(2, 2),
#     nn.Conv2d(64, 64, 3),
#     nn.BatchNorm2d(64, momentum=1, affine=True),
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(2, 2),
#     Flatten(),
#     nn.Linear(64, 4)).to(device)

# print(model)

# model.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/tmp/lymph/thyroid-4x/maml/mt_small_model_lymph/2023-11-22-22-09-20/save_meta_pth/meta_epoch_0.pth'))


input = torch.randn(4, 3, 224, 224).to(device)

target = torch.tensor([1,1,0,1]).to(device)

output = model(input)
criterion = nn.CrossEntropyLoss()
loss = criterion(output,target)

grad = torch.autograd.grad(loss, model.parameters())
print('nihao')


