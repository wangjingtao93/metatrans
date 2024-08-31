import glob
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from copy import deepcopy
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, recall_score, precision_score
from sklearn.metrics import confusion_matrix, classification_report


import torchvision.models
from model.transformer.vit_model import vit_base_patch16_224 
from model.mtb.mtb_model import create_mtb, create_6b_mfc
from model.mtb.mtb_res_bak import create_mtb_res
import timm
from tqdm import tqdm
import os
import model.timm_register_models
from model.ConvNet import Fourlayers 

from model.more_models import *



class dl_comm():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.imagenet_pre_path = '/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result/result_20231010_sub10/pre_train/imagenet'

        self.fig_path = os.path.join(args.store_dir, 'figures') 

        return None
    def _init_net(self):
        

        if self.args.net == 'alexnet':
            self.model = alexnet()

        elif self.args.net == 'convnet_4':
            # self.model = Fourlayers()
            self.model = timm.create_model(self.args.net, num_classes=self.args.num_classes)
        
        # 后续需要加入number_class参数
        elif self.args.net == 'squeezenet1_0':
            # self.model = Fourlayers()
            if self.args.is_load_imagenet:
                self.model = torchvision.models.squeezenet1_0(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.model = torchvision.models.squeezenet1_0(weights=None)
           
            
        elif self.args.net == 'resnet18':
            # self.model = torchvision.models.resnet18(pretrained=self.args.is_load_imagenet)
            if self.args.is_load_imagenet:
                self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = torchvision.models.resnet18(weights=None)
                
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,  self.args.num_classes)

            if self.args.is_load_zk:
                load_path = os.path.join(self.args.project_path,'result_20231010_sub10/zk/dl/resnet18/2023-10-16-19-44-22/best_model.pth')
                self.model.load_state_dict(torch.load(load_path))
                print("Using ****[resnet18]**** load [zk]")
            elif self.args.is_load_imagenet_zk:
                load_path = os.path.join(self.args.project_path, 'result_20231010_sub10/pre_train/zk/dl/resnet18/2023-10-20-15-15-55/meta_epoch/taskid_0/best_model_for_valset_0.pth')
                self.model.load_state_dict(torch.load(load_path))

                print("Using ****[resnet18]**** load [imagenet & zk]")

            elif self.args.is_load_st_sub:

                self.model.load_state_dict(del_fc(self.args.num_classes))

                print("Using ****[resnet18]**** load []")

        elif self.args.net == 'resnet34':
            # self.model = timm.create_model(self.args.net, num_classes=self.args.num_classes)
            if self.args.is_load_imagenet:
                self.model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.model = torchvision.models.resnet34(weights=None)

            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,  self.args.num_classes)

        elif self.args.net == 'resnet50':
            # self.model = timm.create_model(self.args.net, num_classes=self.args.num_classes)
            if self.args.is_load_imagenet:
                self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.model = torchvision.models.resnet50(weights=None)

            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,  self.args.num_classes)

        elif self.args.net == 'vit_base_patch16_224':
            self.model = self.get_vit_base_patch16_224(depth=12)

        elif self.args.net == 'vit_base_patch16_224_depth_6':

            self.model = self.get_vit_base_patch16_224(depth=6)
            print('using transformer with pretrain depth_6')

        elif self.args.net == 'vit_base_patch16_224_depth_3':
            self.model = self.get_vit_base_patch16_224(depth=3)
            print('using transformer with pretrain depth_3')

        elif self.args.net == 'mtb':
            self.model = self.get_mtb()
            print('using mtb with pretrain')          

        elif self.args.net == 'mtb_6b_mfc':
            self.model == self.get_mtb_6b_mfc()
            print('using mtb_6b_mfc with pretrain')       

        elif self.args.net == 'mtb_res':
            self.model = self.get_mtb_res_net()
            print('using **mtb_res** with pretrain')   

        elif self.args.net == 'vit_tiny_patch16_224':
            self.model = timm.create_model(self.args.net, pretrained=self.args.is_load_imagenet, num_classes=self.args.num_classes)
            if self.args.is_load_zk:
                self.model.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/result_20231010_sub10/pre_train/zk/dl/vit_tiny_patch16_224/2023-10-23-17-48-54/meta_epoch/taskid_0/best_model_for_valset_0.pth'))
                print('using **vit_tiny_patch16_224** with zk pretrain')   
        elif self.args.net == 'vit_small_patch16_224':
            self.model = timm.create_model(self.args.net, pretrained=False, num_classes=self.args.num_classes)
            if self.args.is_load_imagenet:
                self.model.load_state_dict(torch.load(glob.glob(os.path.join(self.imagenet_pre_path, self.args.net, '*'))[0]))
                print(f'using **{self.args.net} depth={self.args.trans_depth}**  imagenet Pretrain')  

        elif self.args.net == 'Conformer_tiny_patch16':
            self.model = timm.create_model(self.args.net, pretrained=self.args.is_load_imagenet, num_classes=self.args.num_classes)
            print('using **Conformer_tiny_patch16** with pretrain')   

        elif self.args.net == 'mt_tiny_model_lymph':
            self.model = timm.create_model(self.args.net, pretrained=self.args.is_load_imagenet, num_classes=self.args.num_classes)
            print(f'using **{self.args.net} depth={self.args.trans_depth}** No Pretrain')   
        
        elif self.args.net == 'metatrans':
            self.model = timm.create_model(self.args.net, pretrained=False, num_classes=self.args.num_classes, depth=self.args.trans_depth)
            if self.args.is_load_imagenet:
                # self.mt_lymph_pro_online()
                self.mt_lymph_pro_location()
                print(f'using **{self.args.net} depth={self.args.trans_depth}** imagenet Pretrain')   

        else:
            raise ValueError('No implmentation model')     


        print(f'++++++++++++++using {self.args.net}--------------')

        # print(self.model)
        self.model.train()
        # self.model.cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _init_opt(self):
        self.modellr = 1e-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.modellr)


    def get_vit_base_patch16_224(self, depth):
        model = vit_base_patch16_224(num_classes=self.args.num_classes, depth=depth)
        if self.args.is_load_imagenet:
            # vit_net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=self.args.num_classes)
            vit_net = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=self.args.num_classes)
            vit_net.load_state_dict(torch.load(glob.glob(os.path.join(self.imagenet_pre_path, self.args.net, '*'))[0]))
            copy_trans_params(vit_net, model)
            print(f'Using **[vit] {depth}** load **[imagenet]***')

        elif self.args.is_load_zk:
            if depth == 12:
                load_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/pre_train/result_20231010_sub10/zk/dl/vit_base_patch16_224/2023-10-16-19-49-04/best_model.pth'
            elif depth == 6:
                load_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/imagenet_pre_train/result_20231010_sub10/zk/dl/vit_base_patch16_224_depth_6/2023-10-17-10-31-27/best_model.pth'
            elif depth == 3:
                load_path == '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/pre_train/result_20231010_sub10/imagenet_zk/vit_base_patch16_224_depth_3/2023-10-16-19-22-26/best_model.pth'

            model.load_state_dict(torch.load(load_path))
            print(f'Using **[resnet18] {depth}** load **[zk]***')
        elif self.args.is_load_imagenet_zk:
            if depth == 12:
                load_path = os.path.join(self.args.project_path, 'result_20231010_sub10/pre_train/zk/dl/vit_base_patch16_224/2023-10-20-15-15-45/meta_epoch/taskid_0/best_model_for_valset_0.pth')
            elif depth == 6:
                load_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/imagenet_pre_train/result_20231010_sub10/zk/dl/vit_base_patch16_224_depth_6/2023-10-17-10-31-27/best_model.pth'
            elif depth == 3:
                load_path == '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/pre_train/result_20231010_sub10/imagenet_zk/vit_base_patch16_224_depth_3/2023-10-16-19-22-26/best_model.pth'

            model.load_state_dict(torch.load(load_path))
            print(f'Using **[resnet18] {depth}** load **[imagenet & zk]***')

        return model

    def get_mtb(self):
        model = create_mtb(num_classes=self.args.num_classes, depth=6)
        # model.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/pre_pth/mtb.pth'))
        if self.args.is_load_imagenet:
            vit_net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=self.args.num_classes)
            copy_trans_params(vit_net, model)

        return model
    
    def get_mtb_6b_mfc(self):
        model = create_6b_mfc(num_classes=self.args.num_classes, depth=6)
        # model.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/mtb_6b_mfc_tmp.pth'))
        if self.args.is_load_imagenet:
            vit_net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=self.args.num_classes)
            copy_trans_params(vit_net, model)

        return model

    def get_mtb_res_net(self):
        model = create_mtb_res(num_classes=self.args.num_classes, depth=3)

        if self.args.is_load_zk:
            model.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/imagenet_pre_train/result_20231010_sub10/zk/dl/mtb_res/2023-10-17-11-27-36/best_model.pth'))

        if self.args.is_load_imagenet:
            resnet18 = torchvision.models.resnet18(pretrained=True)
            num_ftrs = resnet18.fc.in_features
            resnet18.fc = nn.Linear(num_ftrs, 4)

            vit_net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=self.args.num_classes)

            copy_trans_params(vit_net, model)
            copy_resnet_params(resnet18, model)


        return model

    def mt_lymph_pro_online(self):
        # 加载transblock预训练参数
        src_model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=self.args.num_classes, depth=self.args.trans_depth)
        src_model.load_state_dict(torch.load(glob.glob(os.path.join(self.imagenet_pre_path, 'vit_small_patch16_224', '*'))[0]))

        src_state_dict = src_model.state_dict()
        dest_state_dict = self.model.state_dict()

        for name, param in src_state_dict.items():
            if name in dest_state_dict:
                dest_state_dict[name].copy_(param)

        # 到底需不需要这一步呢？dest_state_dict是潜拷贝，貌似不需要再load了
        self.model.load_state_dict(dest_state_dict)

    def mt_lymph_pro_location(self):
        path = '/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result/result_20240219/lymph/imagenet/dl/mt_small_model_lymph/2024-02-20-21-58-08/meta_epoch/taskid_0/best_model_for_valset_0.pth'
        src_state_dict = torch.load(path)

        dest_state_dict = self.model.state_dict()

        for name, param in src_state_dict.items():
            if 'meta_cnn_fc' not in name and 'meta_trans_fc' not in name and 'meta_fc' not in name:
                dest_state_dict[name].copy_(param)
            # else:
            #     print('moudle_name: ', name)








    # def __init__(self, device):
    #     self.criterion = nn.CrossEntropyLoss()

    #     self.model = torchvision.models.resnet18(pretrained=False)
    #     num_ftrs = self.model.fc.in_features
    #     self.model.fc = nn.Linear(num_ftrs, 4)

    #     self.model.to(device)

    #     self.modellr = 1e-4
    #     self.optimizer = optim.Adam(self.model.parameters(), lr=self.modellr)

    #     self.device = device

    def train(self, train_loader, epoch):
        self.model.train()
        sum_loss = 0
        total_num = len(train_loader.dataset)
        print(f'train_dataset len: {total_num}', f'train_loader len: {len(train_loader)}')
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data).to(self.device), Variable(target).to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print_loss = loss.data.item()
            sum_loss += print_loss
            if (batch_idx + 1) % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                        100. * (batch_idx + 1) / len(train_loader), loss.item()))
        
        ave_loss = sum_loss / len(train_loader)
        print('epoch:{},loss:{}'.format(epoch, ave_loss))
        return ave_loss


    def val(self, val_loader):
        self.model.eval()

        val_loss = 0

        true_label_list = []
        pred_score_list = []

        total_num = len(val_loader.dataset)
        print(f'val_dataset len: {total_num}' , f'val_loader len: {len(val_loader)}')
        with torch.no_grad():
            for data, target in val_loader:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                pred_score_list.append(output.data.cpu().detach().numpy())
                true_label_list.append(target.cpu().detach().numpy())
                
                print_loss = loss.data.item()
                val_loss += print_loss

            avgloss = val_loss / len(val_loader)
            
        y_true = np.concatenate(true_label_list)
        y_score = np.concatenate(pred_score_list)
        y_pred = np.argmax(y_score, axis=1)

        

        if self.args.num_classes ==2:
            accuracy = accuracy_score(y_true,y_pred)
            precision, recall, f1 = precision_recall_fscore_support(y_true,y_pred,average=self.args.acc_average, zero_division=1)[:-1]
        # 计算 AUC, 绘制ROC 曲线
            fpr, tpr, thresholds = roc_curve(y_true, y_score[:, 1])
            roc_auc = auc(fpr, tpr)

            # 计算sensitivity 和specificity
            cm = confusion_matrix(y_true, y_pred,  labels=[0,1])
            # tn, fp, fn, tp = confusion_matrix(y_true, y_pred,  labels=[0,1]).ravel()
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp +fn)

            print('val++++++++')
            print("accuracy: ", "%.4f"%accuracy)
            print("auc: ","%.4f"%roc_auc)
            print("precision: ","%.4f"%precision)
            print("recall: ","%.4f"%recall)
            print("f1: ","%.4f"%f1)
            print("sensitivity: ", "%.4f"%sensitivity)
            print("specificity: ", "%.4f"%specificity)

            print("average_loss: ","%.4f"%avgloss)
    
            self.model.train()
            return [round(avgloss,4), round(accuracy,4), round(roc_auc,4), round(precision,4), round(recall,4), round(f1,4), round(sensitivity,4), round(specificity,4)], [cm, y_true, y_score,[fpr, tpr]]
        
        if self.args.num_classes > 2:
            labels = []
            for i in self.args.num_classes:
                labels.append(i)
            cm2 = confusion_matrix(y_true,y_pred, labels=labels)
            print(classification_report(y_true, y_pred,target_names=["normal","macro/itc","micro"],digits=2))

            accuracy = accuracy_score(y_true,y_pred)
            f1 = f1_score(y_true,y_pred, labels=labels, average='weighted')
            recall = recall_score(y_true,y_pred, labels=labels, average='weighted')
            precision = precision_score(y_true,y_pred, labels=labels, average='weighted')

            roc_auc = roc_auc_score(y_true,y_pred, labels=labels, average='weighted')

            sensitivity=0
            sensitivity=0

            print('val++++++++')
            print("accuracy: ", "%.4f"%accuracy)
            print("auc: ","%.4f"%roc_auc)
            print("precision: ","%.4f"%precision)
            print("recall: ","%.4f"%recall)
            print("f1: ","%.4f"%f1)
            print("sensitivity: ", "%.4f"%sensitivity)
            print("specificity: ", "%.4f"%specificity)

            print("average_loss: ","%.4f"%avgloss)

            self.model.train()
            return [round(avgloss,4), round(accuracy,4), round(roc_auc,4), round(precision,4), round(recall,4), round(f1,4), round(sensitivity,4), round(specificity,4)]




    def test(self, data_loader, model):
        
        model.eval()
        loss = 0

        true_label_list = []
        pred_score_list = []

        total_num = len(data_loader.dataset)
        print(total_num, len(data_loader))
        with torch.no_grad():
            for data, target in data_loader:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                output = model(data)
                loss = self.criterion(output, target)

                pred_score_list.append(output.data.cpu().detach().numpy())
                true_label_list.append(target.cpu().detach().numpy())

                print_loss = loss.data.item()
                loss += print_loss

        avgloss = loss / len(data_loader)
            
        y_true = np.concatenate(true_label_list)
        y_score = np.concatenate(pred_score_list)
        y_pred = np.argmax(y_score, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1 = precision_recall_fscore_support(y_true, y_pred,average='macro')[:-1]
        
        # 计算 AUC, 绘制ROC 曲线
        if self.args.num_classes > 2:
            fpr, tpr, thresholds = roc_curve(y_true, y_score[:, 1])
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = 0.0
            
        # 计算sensitivity 和specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp +fn)
        
        print('test++++++++++++++++++++++++')
        print("accuracy: ", "%.4f"%accuracy)
        print("auc: ","%.4f"%roc_auc)
        print("precision: ","%.4f"%precision)
        print("recall: ","%.4f"%recall)
        print("f1: ","%.4f"%f1)
        print("sensitivity: ", "%.4f"%sensitivity)
        print("specificity: ", "%.4f"%specificity)
        print("average_loss: ","%.4f"%avgloss)


        
        del model
        
        return [0, round(accuracy,4), round(roc_auc,4), round(precision,4), round(recall,4), round(f1,4), round(sensitivity,4), round(specificity,4)]
    

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        self.modellrnew = self.modellr * (0.1 ** (epoch // 10))# ** 乘方
        print("lr:", self.modellrnew)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.modellrnew    



def copy_trans_params(src_model, dest_model):
    # 获取源模型和目标模型的状态字典
    src_state_dict = src_model.state_dict()
    dest_state_dict = dest_model.state_dict()

    for name, param in src_state_dict.items():
        if name in dest_state_dict:
            dest_state_dict[name].copy_(param)


def copy_resnet_params(src_model, dest_model):
    src_state_dict = src_model.state_dict()
    dest_state_dict = dest_model.state_dict()

    for name, param in src_state_dict.items():
        if 'layer' in name:
           name = 'resdiual_' + name

        if name in dest_state_dict:
            dest_state_dict[name].copy_(param)

# 去除预训练模型的全连接层
def del_fc(num_classes):
    model =  torchvision.models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,  16)
    model.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/result_20231010_sub10/pre_train/st_sub_pretrain/dl/resnet18/2023-11-14-15-33-07/meta_epoch/taskid_0/best_model_for_valset_0.pth'))

    del model.fc
    model.add_module('fc',nn.Linear(num_ftrs,num_classes))
    return model.state_dict()

