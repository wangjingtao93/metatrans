import torch

from torch import nn
from torch import autograd

m = nn.Softmax()
input = autograd.Variable(torch.randn(2, 3))
print(input)
print(m(input))