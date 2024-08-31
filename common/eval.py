import numpy as np
import torch.nn
from sklearn.metrics import confusion_matrix, auc, roc_curve, recall_score, precision_score, f1_score, \
    cohen_kappa_score, accuracy_score,roc_auc_score

def accuracy(output, target, topk=(1,)): #(1,)定义元组中有且仅有一个元素
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) #输出两个tensor分别为值，和索引
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = [] #记录top1,top2,,,topk准确率
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def acc_1(output, target,truelabel=None):
    batch_size=target.size(0)
    _, pred = torch.max(output.data, 1)
    correct = torch.sum(pred == target)
    return correct/batch_size

def allevluate(logits,labels):
    _, pred = torch.max(logits.data, 1)
    acc = accuracy_score(pred, labels)
    fpr,tpr,thresholds=roc_curve(labels,logits[:,1],drop_intermediate=False)
    f1=f1_score(labels,pred)
    try:
        auc=roc_auc_score(labels,logits[:,1])
    except  ValueError:
        auc = None
        pass
    tn, fp, fn, tp = confusion_matrix(labels,pred).ravel()
    sensitivity=tp/(tp+fn)
    specificity=tn/(tn+fp)

    return [acc,auc,sensitivity,specificity,f1,fpr,tpr,tn, fp, fn, tp]

def acc_2(logits,labels):
    _, pred = torch.max(logits.data, 1)
    acc = accuracy_score(pred, labels)
    
    return acc


    

