import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class AttentivePooling(nn.Module):
    def __init__(self, idim, adim, nhead):
        super(AttentivePooling, self).__init__()
        self.nhead = nhead
        self.fc1 = nn.Linear(idim, adim)
        self.fc2 = nn.Linear(adim, nhead, bias=False)

    def forward(self, x, x_len, need_weights=False):
        # create mask
        mask = torch.arange(x.shape[1])[None, :].to(x_len.device) >= x_len[:, None]
        mask = mask.unsqueeze(-1).repeat(1, 1, self.nhead)
        
        # forward
        h = self.fc2(torch.tanh(self.fc1(x)))   # (B, T, nhead)
        h.masked_fill_(mask, float('-inf'))
        att = F.softmax(h, dim=1)
        x = torch.bmm(att.transpose(1, 2), x)   # (B, nhead, idim)
        x = x.view(x.shape[0], -1)

        if need_weights:
            return x, att
        else:
            return x
