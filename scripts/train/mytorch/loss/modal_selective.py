# coding:utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pdb
from mytorch.model.modal_combination import get_modal_combination

IGNORE_INDEX = -1


class ModalSelectiveTrainingLoss(nn.CrossEntropyLoss):
    def __init__(self, lossfunc, lossweight):
        super(ModalSelectiveTrainingLoss, self).__init__()
        self.lossfunc = lossfunc
        self.lossweight = lossweight
        self.comb = get_modal_combination()
        self.gate_loss = MultiLabelSoftMarginLoss()

    def forward(self, y, t, a):
        """
        Args:
            y (torch.Tensor): model output w/ modal combinations (B, n_comb, out_dim)
            t (torch.Tensor): teacher label of the given lossfunc (B, ) / (B, out_dim)
            a (torch.Tensor): gate coefficient of each modal (B, n_modal)
        Returns:
            loss (torch.Tensor): Modal Selective Training Loss := lossfunc + (lossweight * gate_loss)
        """
        losses = torch.stack([self.lossfunc(y[:,i], t) for i in range(y.shape[1])])  # (n_comb, B)
        min_loss, min_index = losses.min(dim=0)
        # gate loss
        gate_loss = self.gate_loss(a, self.comb[min_index].to(a.device))
        # total loss
        loss = min_loss.mean() + self.lossweight * gate_loss
        # selected output (to evaluate train accuracy)
        y_sel = torch.stack([y[i, n, :] for i, n in enumerate(min_index)])
        return loss, y_sel
