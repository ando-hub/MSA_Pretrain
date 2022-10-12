import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


weight = torch.Tensor([0.4073, 1.3558, 0.7760, 1.4609])
x = torch.Tensor([[0.0029, 0.0151, 0.0122, 0.0018],
                  [0.0038, 0.0154, 0.0119, 0.0011],
                  [0.0033, 0.0153, 0.0117, 0.0004],
                  [0.0037, 0.0152, 0.0120, 0.0018],
                  [0.0037, 0.0150, 0.0121, 0.0012],
                  [0.0039, 0.0153, 0.0121, 0.0010],
                  [0.0034, 0.0150, 0.0123, 0.0011],
                  [0.0037, 0.0152, 0.0118, 0.0010]])
ohv = torch.Tensor([[1., 0., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 0., 1.],
                    [1., 0., 0., 0.],
                    [1., 0., 0., 0.],
                    [1., 0., 0., 0.],
                    [1., 0., 0., 0.]])
"""
x = torch.Tensor([[0.0029, 0.0151, 0.0122, 0.0018],
                  [0.0038, 0.0154, 0.0119, 0.0011],
    ])
ohv = torch.Tensor([[1., 0., 0., 0.],
                    [0., 0., 1., 0.],
    ])
"""
t = torch.argmax(ohv, dim=1)


CE = nn.CrossEntropyLoss(weight=weight, reduction='mean')
LogS = nn.LogSoftmax()
NLL = nn.NLLLoss(weight=weight, reduction='mean')

l1 = CE(x, t)
l2 = NLL(LogS(x), t)
l3 = NLL(F.log_softmax(x, dim=1), t)

#l4 = torch.mean(torch.sum(F.log_softmax(x, dim=1) * (ohv*weight/weight.sum()), dim=1))
#l4 = torch.sum(torch.sum(-F.log_softmax(x, dim=1) * (ohv*weight), dim=0)/(weight*x.shape[0]))
l4 = torch.sum(-F.log_softmax(x, dim=1) * (ohv*weight))/torch.sum(ohv*weight)

print(l1, l2, l3, l4)

# no weight
CEno = nn.CrossEntropyLoss()
NLLno = nn.NLLLoss()

l1no = CEno(x, t)
l2no = NLLno(F.log_softmax(x, dim=1), t)
#l4no = - torch.mean(torch.sum(F.log_softmax(x, dim=1) * (ohv), dim=1))
#l4no = torch.sum(torch.sum(-F.log_softmax(x, dim=1) * (ohv), dim=0)/(x.shape[0]))
l4no = torch.sum(-F.log_softmax(x, dim=1) * (ohv))/torch.sum(ohv)

print(l1no, l2no, l4no)

pdb.set_trace()

