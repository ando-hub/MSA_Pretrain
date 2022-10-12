# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F
import pdb

mlloss = nn.MultiLabelSoftMarginLoss()
bcelogitsloss = nn.BCEWithLogitsLoss()
bceloss = nn.BCELoss()
#kldivloss = nn.KLDivLoss(reduction='sum')
kldivloss = nn.KLDivLoss(reduction='batchmean')
#kldivlossmean = nn.KLDivLoss(reduction='mean')
sigmoid = nn.Sigmoid()


data = torch.FloatTensor([[-10, 0, 10, 0, -10], [10, 0, 0, -10, -10]])
label = torch.FloatTensor([[0,0.5,1,0.5,0], [1,0.5,0.5,0,0]])
#label = torch.FloatTensor([[0,0.5,1,0.5,0], [1,0.5,0.5,0,0]])
#hard_label = torch.LongTensor([2, 0])

#data = torch.FloatTensor([[-10, 0, 10, 0, -10]])
#label = torch.FloatTensor([[0,0.5,1,0.5,0]])
#data = torch.FloatTensor([[0.36, 0.48, 0.16], [0.36, 0.48, 0.16]])
#data = torch.FloatTensor([[0.36, 0.48, 0.16]])
#label = torch.FloatTensor([[0.30, 0.50, 0.20], [0.33, 0.33, 0.33]])
#label = torch.FloatTensor([[0.30, 0.50, 0.20]])
#label = torch.FloatTensor([[0.33, 0.33, 0.33]])

#print(mlloss(data, label))
#print(bcelogitsloss(data, label))
#print(bceloss(sigmoid(data), label))
#print(kldivloss(sigmoid(data), label))
#print(kldivloss(data.log(), label))
#print(F.kl_div(data.log(), label, None, None, 'batchmean'))
#print((label*(label/data).log()).sum(dim=1).mean())
#print(kldivloss(F.log_softmax(data, dim=1), label))
#print(kldivloss(F.log_softmax(data, dim=1), label))
#print(kldivlossmean(F.log_softmax(data, dim=1), label))
#print(F.softmax(data))
print(F.softmax(data, dim=1))
print(F.softmax(data, dim=1).log())
print(F.log_softmax(data))
print(F.log_softmax(data, dim=0))
print(F.log_softmax(data, dim=1))
#print(label)
#pdb.set_trace()
