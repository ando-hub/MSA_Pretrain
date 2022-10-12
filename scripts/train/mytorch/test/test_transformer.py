import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import copy

x = torch.rand(10, 2, 4)    # nlen, nbat, ndim
x_len = torch.tensor([10, 5]).long()
z = x.clone()
z[5:, 1, :] = 0.

encl = nn.TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=128)
model = nn.TransformerEncoder(encl, num_layers=4)

max_len = x_len.max().item()
mask = torch.arange(max_len)[None, :] >= x_len[:, None]

#y = model(x)
m1 = copy.deepcopy(model)
m2 = copy.deepcopy(model)
m1.eval()
m2.eval()
y = m1(x, src_key_padding_mask=mask)
w = m2(z, src_key_padding_mask=mask)

pdb.set_trace()

