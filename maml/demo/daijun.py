import os

import higher
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from model import resnet_model
from dataset.fourset  import MetaDataSet
from dataset.allDeviceTwoClassPvrlSet import MetaDataSet as MetaDataTestSet
import tqdm
import os.path as osp
from utils import evaluate
from sklearn.metrics import confusion_matrix, auc, roc_curve, recall_score, precision_score, f1_score, cohen_kappa_score,accuracy_score
class Trainer:
    def __init__(self,args):
        self.args=args
        self.epoch=args.epoch
        self.train_dataset=MetaDataSet(args.root, 'train', args.batchsz, args.n_way, args.k_shot, args.k_query, args.resize)
        self.train_dataloader=DataLoader(self.train_dataset,args.meta_batch,shuffle=True,drop_last=True)
        self.val_dataset = MetaDataSet(args.root, 'val', args.batchsz, args.n_way, args.k_shot, args.k_query, args.resize)
        self.val_dataloader = DataLoader(self.train_dataset, args.meta_batch, shuffle=True, drop_last=True)
        self.test_dataset = MetaDataTestSet(args.root, 'test', args.batchsz, args.n_way, args.k_shot, args.k_query, args.resize)
        self.test_dataloader = DataLoader(self.test_dataset, args.meta_batch, shuffle=True, drop_last=True)
        self.model=resnet_model.resnet18(args.n_way).cuda()
        self.critirion=torch.nn.CrossEntropyLoss()
        self.optimize=torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.save_path = os.path.join(args.model_root, args.exp_label) + '/'
        self.modelname = args.exp_label + 'maxacc'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
    def train(self):
        self.model.train()

        CD = self.critirion
        best_epoch=0
        best_acc=0
        for epoch in range(self.epoch+1):
            tq = tqdm.tqdm(self.train_dataloader)
            for batch_idx, batch in enumerate(tq, 1):
                support_x, support_y, query_x, query_y = [_.cuda() for _ in batch]
                task_num, setsz, c_, h, w = support_x.size()

                # Initialize the inner optimizer to adapt the parameters to
                # the support set.
                # 每一个epoch都实例化一个，不用从上一个epoch开始
                inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.args.inner_lr)
                qry_losses = []
                train_qry_acc_list=[]
                self.optimize.zero_grad()
                for i in range(task_num):
                    with higher.innerloop_ctx(self.model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                        # Optimize the likelihood of the support set by taking
                        # gradient steps w.r.t. the model's parameters.
                        # This adapts the model's meta-parameters to the task.
                        # higher is able to automatically keep copies of
                        # your network's parameters as they are being updated.
                        for _ in range(self.args.n_inner_updates):
                            spt_logits = fnet(support_x[i])
                            spt_loss = CD(spt_logits, support_y[i])
                            diffopt.step(spt_loss)

                        # The final set of adapted parameters will induce some
                        # final loss and accuracy on the query dataset.
                        # These will be used to update the model's meta-parameters.
                        qry_logits = fnet(query_x[i])
                        qry_loss = CD(qry_logits, query_y[i])
                        qry_losses.append(qry_loss.detach().cpu())
                        acc=evaluate.acc(qry_logits.detach().cpu(), query_y[i].detach().cpu())
                        train_qry_acc_list.append(acc)

                        # Update the model's meta-parameters to optimize the query
                        # losses across all of the tasks sampled in this batch.
                        # This unrolls through the gradient steps.
                        qry_loss.backward()  # 为什么要在这呢
                self.optimize.step()
                tq.set_description("epoch {},queryloss {:.4f},acc {:.4f}".format(epoch,np.mean(np.array(qry_losses)),np.mean(train_qry_acc_list)))
            "-----------------------------------------------------------------------------------------------------------"
            CD = self.critirion

            val_qry_losses = []
            val_acc_list=[]

            for batch_idx, batch in enumerate(self.val_dataloader, 1):
                support_x, support_y, query_x, query_y = [_.cuda() for _ in batch]
                task_num, setsz, c_, h, w = support_x.size()

                n_inner_iter = self.args.n_inner_updates
                inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.args.inner_lr)

                for i in range(task_num):
                    with higher.innerloop_ctx(self.model, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                        # Optimize the likelihood of the support set by taking
                        # gradient steps w.r.t. the model's parameters.
                        # This adapts the model's meta-parameters to the task.
                        for _ in range(n_inner_iter):
                            spt_logits = fnet(support_x[i])
                            spt_loss = CD(spt_logits, support_y[i])
                            diffopt.step(spt_loss)

                        # The query loss and acc induced by these parameters.
                        qry_logits = fnet(query_x[i]).detach().cpu()
                        qry_loss = CD(qry_logits, query_y[i].detach().cpu())
                        val_acc=evaluate.acc(qry_logits,query_y[i].cpu())
                        val_acc_list.append(val_acc)
                        val_qry_losses.append(qry_loss.detach().cpu())
            if np.mean(np.array(val_acc_list))>best_acc:
                best_epoch=epoch
                best_acc=np.mean(val_acc_list)
                self.save_model(self.modelname)
            if epoch%5==0:
                self.save_model('epoch{}'.format(epoch))
            print("epoch {} val_acc {:.4f} best_epoch {} best_acc {:.4f}".format(epoch,np.mean(val_acc_list),best_epoch,best_acc))
    def test(self):
        self.model.load_state_dict(torch.load(osp.join(self.save_path, self.modelname + '.pth'))['params'])
        self.model.train()
        CD = self.critirion
        qry_acc_list = []
        class_one_recall = []
        class_two_recall = []
        class_three_recall = []
        class_four_recall = []
        tq= tqdm.tqdm(self.test_dataloader)
        for batch_idx, batch in enumerate(tq, 1):
            support_x, support_y, query_x, query_y = [_.cuda() for _ in batch]
            task_num, setsz, c_, h, w = support_x.size()

            n_inner_iter = self.args.n_inner_updates
            inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.args.inner_lr)

            for i in range(task_num):
                with higher.innerloop_ctx(self.model, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                    # Optimize the likelihood of the support set by taking
                    # gradient steps w.r.t. the model's parameters.
                    # This adapts the model's meta-parameters to the task.
                    for _ in range(n_inner_iter):
                        spt_logits = fnet(support_x[i])
                        spt_loss = CD(spt_logits, support_y[i])
                        diffopt.step(spt_loss)

                    # The query loss and acc induced by these parameters.
                    qry_logits = fnet(query_x[i])
                    logits = qry_logits.detach().cpu()
                    _, pred = torch.max(logits.data, 1)
                    labels = query_y[i].detach().cpu()
                    acc = accuracy_score(pred, labels)
                    Recall = recall_score(pred, labels, average=None)
                    class_one_recall.append(Recall[0])
                    class_two_recall.append(Recall[1])
                    class_three_recall.append(Recall[2])
                    class_four_recall.append(Recall[3])
                    qry_acc_list.append(acc)

        print("test acc{:.4f} test accstd{:.4f}  one recall{:.4f} two recall{:.4f} three recall{:.4f} four recall{:.4f}".format(np.mean(qry_acc_list), np.std(qry_acc_list), np.mean(class_one_recall),np.mean(class_two_recall), np.mean(class_three_recall), np.mean(class_four_recall)))

    def save_model(self,name):
        torch.save(dict(params=self.model.state_dict()), osp.join(self.save_path, name + '.pth'))