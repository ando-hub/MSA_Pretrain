import torch
import copy
from mytorch.optimizer.noam import NoamOpt


def get_optimizer_state(optim):
    if isinstance(optim, NoamOpt):
        _step, _sd = optim.get_state()
        sd = copy.deepcopy(_sd)
        for s, _d in sd['state'].items():
            for k, v in _d.items():
                if isinstance(v, torch.Tensor):
                    sd['state'][s][k] = v.cpu()
        return (_step, sd)
    else:
        _sd = optim.state_dict()
        sd = copy.deepcopy(_sd)
        for s, _d in sd['state'].items():
            for k, v in _d.items():
                if isinstance(v, torch.Tensor):
                    sd['state'][s][k] = v.cpu()
        return sd


def set_optimizer_state(optim, state, device):
    if isinstance(optim, NoamOpt):
        step, sd = state
        for s, _d in sd['state'].items():
            for k, v in _d.items():
                if isinstance(v, torch.Tensor):
                    sd['state'][s][k] = v.to(device)
        optim.set_state((step, sd))
    else:
        sd = state
        for s, _d in sd['state'].items():
            for k, v in _d.items():
                if isinstance(v, torch.Tensor):
                    sd['state'][s][k] = v.to(device)
        optim.load_state_dict(sd)
