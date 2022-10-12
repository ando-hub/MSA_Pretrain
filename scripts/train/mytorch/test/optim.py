# coding:utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pdb import set_trace


def get_optimizer(model, optimization, lr, lrstep, warmup_ep=-1):
    if optimization == 'Adam':
        optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr
                )
    elif optimization == 'momentumSGD':
        optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, momentum=0.9, weight_decay=0.0001
                )
    elif optimization == 'AdamW':
        optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=0.01
                )

    if lrstep:
        min_lr = lr/16.        # 1/16 = (1/2) ** 5
        __scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=2, verbose=True, min_lr=min_lr
                )
        if warmup_ep > 1:
            scheduler = WarmupScheduler(
                    optimizer, __scheduler, init_lr=0., goal_ep=warmup_ep
                    )
        else:
            scheduler = __scheduler
    else:
        scheduler = None
    return optimizer, scheduler


class WarmupScheduler():
    """ warm-up + ReduceLROnPlateau scheduler.
    warmup is applied in every epoch, not in every iteration.

    Args:
        optimizer: Wrapped optimizer
        after_scheduler: after warmup, use this scheduler (ReduceLROnPlateau only)
        init_lr: initial learning rate
        goal_ep: goal ep of warm-up step
        verbose: show lr update
    """
    def __init__(self, optimizer, after_scheduler=None, init_lr=0., goal_ep=5, verbose=True):
        if type(after_scheduler) != ReduceLROnPlateau:
            raise NotImplementedError('WarmupScheduler supports only ReduceLROnPlateau')
        if goal_ep <= 1:
            raise ValueError('warmup goal_ep must be > 2 !')
        self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))

        self.optimizer = optimizer
        self.after_scheduler = after_scheduler
        self.init_lr = init_lr
        self.goal_ep = goal_ep
        self.verbose = verbose
        self.next_ep = 0
        self.min_lrs = after_scheduler.min_lrs
        self.step(None)

    def _set_lr(self, lrs):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lrs[i]
            if self.verbose:
                print('Epoch {:5d}: set learning rate '
                      'of group {} to {:.4e}'.format(self.next_ep, i, lrs[i]))

    def step(self, metrics):
        self.next_ep += 1
        if self.next_ep <= self.goal_ep:
            lrs = [self.init_lr+(base_lr-self.init_lr)*(self.next_ep/self.goal_ep) for base_lr in self.base_lrs]
            self._set_lr(lrs)
        else:
            self.after_scheduler.step(metrics)
