# coding:utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pdb
from mytorch.model.modal_combination import get_modal_combination

IGNORE_INDEX = -1


class MultiLabelSoftMarginLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, ignore_index=IGNORE_INDEX, reduction='mean'):
        super(MultiLabelSoftMarginLoss, self).__init__(
                weight=weight, ignore_index=ignore_index, reduction=reduction
                )

    def forward(self, y, t):
        if (t == self.ignore_index).sum().item():
            raise NotImplementedError(
                    'Do not use ignore_index ({}) in the label!'.format(self.ignore_index)
                    )
        _y = torch.sigmoid(y)
        _lp = t * torch.log(_y)
        _lm = (1-t) * torch.log(1-_y)
        if self.weight is not None:
            _lp = self.weight[1] * _lp
            _lm = self.weight[0] * _lm
        _l = _lp+_lm
        if self.reduction == 'mean':
            return -1*torch.mean(_l)
        elif self.reduction == 'none':
            return -1*torch.mean(_l, dim=1)
