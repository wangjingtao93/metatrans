import torch.nn.functional as F
import torch.nn as nn


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        
    def forward(self, p, q):
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        return loss
