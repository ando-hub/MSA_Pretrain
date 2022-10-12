import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


x = torch.rand(2,3,4)

bn = nn.BatchNorm1d(3)
y = bn(x)

z = torch.rand(6,4)
bn2 = nn.BatchNorm1d(4)
w = bn2(z)

pdb.set_trace()

