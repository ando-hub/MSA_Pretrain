import torch
import torch.nn as nn
import pdb

from blocks import TDNNBlock

class mymodel(nn.Module):
    def __init__(
            self, nin, nout, context, batchnorm=False, activ=None, drop=0.0):
        super(mymodel, self).__init__()
        self.l1 = TDNNBlock(nin, nout, context, batchnorm, activ, drop)
        pdb.set_trace()

    def forward(self, x, x_len):
        return self.l1(x, x_len)


x = torch.rand(2, 10, 4)
x_len = torch.tensor([10, 5]).long()

model = mymodel(4, 3, (-1, 0, 1), False, None, 0.5)
#model = mymodel(4, 3, (0,), False, None, 0.)

y, y_len = model(x, x_len)
pdb.set_trace()


