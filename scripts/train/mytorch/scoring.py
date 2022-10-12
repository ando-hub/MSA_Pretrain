import torch
import torch.nn.functional as F
import numpy as np
import pdb


def get_score(y, t, classification_type):
    assert classification_type in ['binary', 'multiclass', 'regress']
    # classification_type = 'binary' if len(t.shape) > 1 else 'multiclass'

    if classification_type == 'binary':
        match = (torch.sigmoid(y) > 0.5) == t
        total_elem = torch.numel(match)
        total_match = match.sum().cpu().detach().item()
        label_match = match.sum(dim=0).cpu().detach().numpy()
    elif classification_type == 'multiclass':
        match = torch.argmax(y, dim=1) == t
        total_elem = torch.numel(match)
        total_match = match.sum().cpu().detach().item()
        label_match = np.array([total_match])
    elif classification_type == 'regress':
        # MOSI/MOSEI sentiment binary classification (y>0 or y<0)
        nonzero_idx = (t != 0)
        _t = t[nonzero_idx] > 0
        _y = y[nonzero_idx] > 0
        match = (_t == _y)
        total_elem = torch.numel(match)
        total_match = match.sum().cpu().detach().item()
        label_match = np.atleast_1d(match.sum(dim=0).cpu().detach().numpy())
    return total_elem, total_match, label_match


def get_total_score(scores):
    total_elems, total_matches, label_matches = zip(*scores)
    total_matches = sum(total_matches)
    total_elems = sum(total_elems)
    label_matches = np.stack(label_matches).sum(axis=0)
    label_elems = total_elems/label_matches.size
    res = {}
    res['score/accuracy_total'] = total_matches/total_elems
    for i, m in enumerate(label_matches):
        res['score/accuracy_class{}'.format(i)] = m/label_elems
    return res

