import torch
from model.global_average_pooling import GlobalAveragePooling
import pdb

gap = GlobalAveragePooling()

x = torch.tensor(
        [[[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]],
        [[0, 1, 2],
            [3, 4, 5],
            [0, 0, 0]]]
        )

x_len = torch.tensor([3, 2]).long()

y = gap(x, x_len)

pdb.set_trace()
