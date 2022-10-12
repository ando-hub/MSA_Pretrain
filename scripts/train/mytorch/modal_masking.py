import numpy as np
import torch
import random
import pdb

random.seed(777)


class ModalMasking:
    def __init__(self, ep_start_2modal, ep_end_2modal, ep_start_3modal, ep_end_3modal):
        assert ep_start_2modal <= ep_start_3modal
        assert ep_end_2modal <= ep_end_3modal
        self.ep_start_2modal = ep_start_2modal
        self.ep_end_2modal = ep_end_2modal
        self.ep_start_3modal = ep_start_3modal
        self.ep_end_3modal = ep_end_3modal
        self.modal_table = (
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (1, 0, 1),
                (0, 1, 1),
                (1, 1, 1)
                )
        self.weights = None

    def set_epoch(self, ep):
        prob_2modal = (ep-self.ep_start_2modal)/(self.ep_end_2modal-self.ep_start_2modal)
        prob_2modal = min(1, max(prob_2modal, 0))
        prob_3modal = (ep-self.ep_start_3modal)/(self.ep_end_3modal-self.ep_start_3modal)
        prob_3modal = min(1, max(prob_3modal, 0))
        
        weight_1modal = 1-prob_2modal
        weight_2modal = prob_2modal - prob_3modal
        weight_3modal = prob_3modal
        self.weights = (
                weight_1modal/3, weight_1modal/3, weight_1modal/3,
                weight_2modal/3, weight_2modal/3, weight_2modal/3,
                weight_3modal
                )

    def __call__(self, x):
        assert self.weights, 'call .set_epoch(ep) at first'
        n_batch = x[0].shape[0]
        select_modal = torch.Tensor(
                random.choices(self.modal_table, k=n_batch, weights=self.weights)
                ).bool()
        if x[0] is not None:
            x[0][~select_modal[:, 0], :, :] = 0
        if x[2] is not None:
            x[2][~select_modal[:, 1], :, :] = 0
        if x[4] is not None:
            x[4][~select_modal[:, 2], :, :] = 0
        return x
