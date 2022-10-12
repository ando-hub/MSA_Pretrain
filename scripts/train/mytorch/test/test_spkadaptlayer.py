import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

class AdaptBlock(nn.Module):
    def __init__(self, nin, nout, nhead):
        super(AdaptBlock, self).__init__()
        ld = {str(i): nn.Linear(nin, nout) for i in range(nhead)}
        self.linears = nn.ModuleDict(ld)
        self.nhead = nhead
        self.nout = nout
    
    def forward(self, x, alpha):
        """
        x    : [nbat x ndim]  input vector
        alpha: [nbat x nhead] head weight
        """
        nbat, ndim = x.shape
        y = torch.zeros(
                (nbat, self.nout),
                dtype=torch.float32, 
                device=x.device
                )
        for i in range(self.nhead):
            _a = alpha[:, i]
            _l = self.linears[str(i)](x)
            y += (_l.T * _a).T
        return y
        

if __name__ == '__main__':
    x = torch.rand(3, 3)
    model = AdaptBlock(3, 4, 2)
    #alpha = torch.FloatTensor([0.2, 0.8])
    alpha = torch.FloatTensor([[1., 0.], [0.5, 0.5], [0, 1]])
    
    y = model(x, alpha)
    
    #model = Linears()
    #y = model(x)
    
    pdb.set_trace()
