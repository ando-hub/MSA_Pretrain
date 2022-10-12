import torch
import torch.nn as nn
from pdb import set_trace

naux = 2
nbat = 2
nin = 3
nhid = 4

x = torch.rand((nbat, nin))
t = torch.tensor([[0, 1, 1], [-1, 0, 1]])
aux = torch.tensor([[0, 1], [1, 0]])

layerdict = nn.ModuleDict({str(i): nn.Linear(nin, nhid) for i in range(naux)})

y = torch.zeros(nbat, nhid)
for i, j in enumerate(aux.argmax(1)):
    y[i] = layerdict[str(int(j))](x[i])

set_trace()
