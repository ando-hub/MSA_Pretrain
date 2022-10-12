import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from mytorch.model.attentive_pooling import AttentivePooling
from mytorch.model.gate_pooling import GatePooling
from mytorch.model.modal_combination import get_modal_combination


class MultiModalDecoder(nn.Module):
    def __init__(self, in_dim, dec_dim, dec_layer, out_dim, nhead=8, dropout_rate=0.1, pooling='ap', lossfunc='ce'):
        super(MultiModalDecoder, self).__init__()
        self.pooling = pooling
        if pooling == 'sap':
            self.attn = AttentivePooling(in_dim, in_dim, nhead)
            attn_dim = in_dim * nhead
        elif pooling == 'gate':
            self.attn = GatePooling(in_dim, 3)
            attn_dim = in_dim
        elif pooling == 'concat':
            attn_dim = in_dim*3
        else:
            raise ValueError('pooling must be either sap, gate or concat')

        decoder = []
        for i in range(dec_layer):
            _in_dim = attn_dim if i == 0  else dec_dim
            _out_dim = out_dim if i == dec_layer-1 else dec_dim
            decoder.append(nn.Linear(_in_dim, _out_dim))
            if i != dec_layer-1:
                decoder.append(nn.LayerNorm(_out_dim))
                decoder.append(nn.ReLU())
                decoder.append(nn.Dropout(dropout_rate))
        self.dec = nn.Sequential(*decoder)

        self.modal_selective_training = True if lossfunc == 'most' else False
        if self.modal_selective_training:
            self.comb = get_modal_combination()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor (B, 3, embed_dim)

        Returns:
            (torch.Tensor): output embedding (B, out_dim)
        """
        if self.pooling == 'sap':
            x, att = self.attn(x, torch.ones(x.shape[0], device=x.device)*3, need_weights=True)
        elif self.pooling == 'gate':
            if self.training and self.modal_selective_training:
                given_weights = self.comb.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.device)
            else:
                given_weights = None
            x, att = self.attn(x, need_weights=True, given_weights=given_weights)
        elif self.pooling == 'concat':
            x = x.view(x.shape[0], -1)
            att = None
        x = self.dec(x)
        return x, att
