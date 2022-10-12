import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Linears(nn.Module):
    def __init__(self):
        super(Linears, self).__init__()
        layers = [
                nn.Linear(3, 4),
                nn.Tanh(),
                nn.Linear(4, 5),
                ]
        # add layerdict
        ld = {str(i): nn.Linear(5, 6) for i in range(3)}
        layers.append(nn.ModuleDict(ld))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        #return self.layers(x)
        
        for i, l in enumerate(self.layers):
            print(i, l)
            if isinstance(l, nn.ModuleDict):
                x = l[str(0)](x)
            else:
                x = l(x)
        return x

if __name__ == '__main__':
    x = torch.rand(2, 4, 3)
    
    model = Linears()
    y = model(x)
    
    pdb.set_trace()
