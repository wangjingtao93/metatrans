from networkx import selfloop_edges
import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, auc, roc_curve, recall_score, precision_score, f1_score, cohen_kappa_score,accuracy_score

from common.meta.gbml import GBML

class Learner(GBML):
    """
    This is a learner class, which will accept a specific network module, such as OmniNet that define the network forward
    process. Learner class will create two same network, one as theta network and the other acts as theta_pi network.
    for each episode, the theta_pi network will copy its initial parameters from theta network and update several steps
    by meta-train set and then calculate its loss on meta-test set. All loss on meta-test set will be sumed together and
    then backprop on theta network, which should be done on metalaerner class.
    For learner class, it will be responsible for update for several steps on meta-train set and return with the loss on
    meta-test set.
    """

    def __init__(self, args):
        """
        It will receive a class: net_cls and its parameters: args for net_cls.
        :param net_cls: class, not instance
        :param args: the parameters for net_cls
        """
        super().__init__(args)
        self.args = args
        self._init_net()
        self._init_opt()
	
        # we will create two class instance meanwhile and use one as theta network and the other as theta_pi network.
        # you must call create_pi_net to create pi network additionally
        # self.net_pi = net_cls(*args)
        self.network_pi = deepcopy(self.network)
        for name, param in self.network_pi.named_parameters():
            param.requires_grad = True

        # update theta_pi = theta_pi - lr * grad
        # according to the paper, here we use naive version of SGD to update theta_pi
        # 0.1 here means the learner_lr
        self.optimizer = optim.SGD(self.network_pi.parameters(), self.args.inner_lr)
        self.outer_criterion = nn.CrossEntropyLoss()
        self.inner_criterion = nn.CrossEntropyLoss()

    def parameters(self):
        """
        Override this function to return only net parameters for MetaLearner's optimize
        it will ignore theta_pi network parameters.
        :return:
        """
        return self.network.parameters()

    def update_pi(self):
        """
        copy parameters from self.net -> self.net_pi
        :return:
        """
        for m_from, m_to in zip(self.network.modules(), self.network_pi.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def inner_loop(self, support_x, support_y, query_x, query_y, num_updates):
        """
        learn on current episode meta-train: support_x & support_y and then calculate loss on meta-test set: query_x&y
        :param support_x: [setsz, c_, h, w]
        :param support_y: [setsz]
        :param query_x:   [querysz, c_, h, w]
        :param query_y:   [querysz]
        :param num_updates: 5
        :return:
        """
        # now try to fine-tune from current $theta$ parameters -> $theta_pi$
        # after num_updates of fine-tune, we will get a good theta_pi parameters so that it will retain satisfying
        # performance on specific task, that's, current episode.
        # firstly, copy theta_pi from theta network
        self.update_pi()

        # update for several steps
        for i in range(num_updates):
            # forward and backward to update net_pi grad.
            logits = self.network_pi(support_x)
            loss = self.inner_criterion(logits, support_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Compute the meta gradient and return it, the gradient is from one episode
        # in metalearner, it will merge all loss from different episode and sum over it.
        qry_logits = self.network_pi(query_x)
        loss = self.inner_criterion(qry_logits,query_y)

        logits = qry_logits.detach().cpu()
        labels = query_y.detach().cpu()
        _, pred = torch.max(logits.data, 1)
        acc = accuracy_score(pred, labels)

        # gradient for validation on theta_pi
        # after call autorad.grad, you can not call backward again except for setting create_graph = True，为啥要计算二阶导
        # as we will use the loss as dummpy loss to conduct a dummy backprop to write our gradients to theta network,
        # here we set create_graph to true to support second time backward.为啥要计算二阶导
        grads_pi = autograd.grad(loss, self.network_pi.parameters(), create_graph=True) 

        return loss, grads_pi, acc

    def net_forward(self, support_x, support_y):
        """
        This function is purely for updating net network. In metalearner, we need the get the loss op from net network
        to write our merged gradients into net network, hence will call this function to get a dummy loss op.
        :param support_x: [setsz, c, h, w]
        :param support_y: [sessz, c, h, w]
        :return: dummy loss and dummy pred
        """
        pred = self.network(support_x)
        loss = self.outer_criterion(pred, support_y)
        return loss, pred


class Reptile(GBML):
    """
    As we have mentioned in Learner class, the metalearner class will receive a series of loss on different tasks/episodes
    on theta_pi network, and it will merage all loss and then sum over it. The summed loss will be backproped on theta
    network to update theta parameters, which is the initialization point we want to find.
    """

    def __init__(self, args):
        super().__init__(args)
        # it will contains a learner class to learn on episodes and gather the loss together.
        self.learner = Learner(args)
        self.args = args
        # the optimizer is to update theta parameters, not theta_pi parameters. 因为重写了parameters函数，只返回self.network参数
        self.optimizer = optim.Adam(self.learner.parameters(), lr=self.args.outer_lr)
        # self.optimizer = optim.Adam(self.learner.network.parameters(), lr=self.args.outer_lr)

    def train(self, db, epoch):
        tqdm_train = tqdm(db)
        qry_acc_all_tasks = []
        qry_loss_all_tasks = []
        for batch_idx, batch in enumerate(tqdm_train, 1):
            support_x, support_y, query_x, query_y = unpack_batch(batch)
            acc, loss = self.eposide(support_x, support_y, query_x, query_y)
            tqdm_train.set_description('Meta Training_Tasks Epoch {}, batch_idx {}, acc={:.4f}, Loss={:.4f}'.format(epoch, batch_idx, acc, loss))	
			
            qry_acc_all_tasks.append(acc)
            qry_loss_all_tasks.append(loss.item())
        
        ave_qry_acc_all_tasks = round(np.array(qry_acc_all_tasks).mean(), 4) 
        ave_qry_loss_all_tasks = round(np.array(qry_loss_all_tasks).mean(), 4)

        return ave_qry_acc_all_tasks, round(qry_acc_all_tasks[-1].item(),4), ave_qry_loss_all_tasks

    def write_grads(self, dummy_loss, sum_grads_pi):
        """
        write loss into learner.net, gradients come from sum_grads_pi.
        Since the gradients info is not calculated by general backward, we need this function to write the right gradients
        into theta network and update theta parameters as wished.
        :param dummy_loss: dummy loss, nothing but to write our gradients by hook
        :param sum_grads_pi: the summed gradients
        :return:
        """

        # Register a hook on each parameter in the net that replaces the current dummy grad
        # with our grads accumulated across the meta-batch
        hooks = []

        for i, v in enumerate(self.learner.parameters()):
            def closure():
                ii = i
                return lambda grad: sum_grads_pi[ii]

            # if you write: hooks.append( v.register_hook(lambda grad : sum_grads_pi[i]) )
            # it will pop an ERROR, i don't know why?
            if v.requires_grad:
                hooks.append(v.register_hook(closure()))

        # use our sumed gradients_pi to update the theta/net network,
        # since our optimizer receive the self.net.parameters() only.
        self.optimizer.zero_grad()
        dummy_loss.backward()
        self.optimizer.step()

        # if you do NOT remove the hook, the GPU memory will expode!!!
        for h in hooks:
            h.remove()

    def eposide(self, support_x, support_y, query_x, query_y):
        """
        Here we receive a series of episode, each episode will be learned by learner and get a loss on parameters theta.
        we gather the loss and sum all the loss and then update theta network.
        setsz = n_way * k_shotf
        querysz = n_way * k_shot
        :param support_x: [meta_batchsz, setsz, c_, h, w]
        :param support_y: [meta_batchsz, setsz]
        :param query_x:   [meta_batchsz, querysz, c_, h, w]
        :param query_y:   [meta_batchsz, querysz]
        :return:
        """
        sum_grads_pi = None
        meta_batchsz = support_y.size(0)

        # support_x[i]: [setsz, c_, h, w]
        # we do different learning task sequentially, not parallel.，一个任务一个任务的计算，而不是一个batch任务的计算
        accs = []
        # for each task/episode.
        for i in range(meta_batchsz):
            _, grad_pi, episode_acc = self.learner.inner_loop(support_x[i], support_y[i], query_x[i], query_y[i], self.args.n_inner)
            accs.append(episode_acc)
            if sum_grads_pi is None:
                sum_grads_pi = grad_pi
            else:  # accumulate all gradients from different episode learner
                sum_grads_pi = [torch.add(i, j) for i, j in zip(sum_grads_pi, grad_pi)]

        # As we already have the grads to update
        # We use a dummy forward / backward pass to get the correct grads into self.net
        # the right grads will be updated by hook, ignoring backward.
        # use hook mechnism to write sumed gradient into network.
        # we need to update the theta/net network, we need a op from net network, so we call self.learner.net_forward
        # to get the op from net network, since the loss from self.learner.forward will return loss from net_pi network.
        dummy_loss, _ = self.learner.net_forward(support_x[0], support_y[0])
        self.write_grads(dummy_loss, sum_grads_pi)

        return round(np.array(accs).mean(),6), dummy_loss

    def pred(self, support_x, support_y, query_x, query_y):
        """
        predict for query_x
        :param support_x:
        :param support_y:
        :param query_x:
        :param query_y:
        :return:
        """
        meta_batchsz = support_y.size(0)

        accs = []
        # for each task/episode.
        # the learner will copy parameters from current theta network and then fine-tune on support set.
        for i in range(meta_batchsz):
            _, _, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
            accs.append(episode_acc)

        return np.array(accs).mean()
	
def unpack_batch(batch):

    device = torch.device('cuda')
    train_inputs, train_targets = batch[0]
    train_inputs = train_inputs.to(device=device)
    train_targets = train_targets.to(device=device, dtype=torch.long)

    test_inputs, test_targets = batch[1]
    test_inputs = test_inputs.to(device=device)
    test_targets = test_targets.to(device=device, dtype=torch.long)

    return train_inputs, train_targets, test_inputs, test_targets 