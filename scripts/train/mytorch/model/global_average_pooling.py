import torch
import torch.nn as nn


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x, x_len):
        # create mask
        mask = torch.arange(x.shape[1])[None, :].to(x_len.device) >= x_len[:, None] # 2-D mask
        mask = mask.unsqueeze(-1).repeat(1, 1, x.shape[2])
        
        x.masked_fill_(mask, 0)
        return x.sum(dim=1) / x_len.unsqueeze(-1).expand(-1, x.shape[-1])
