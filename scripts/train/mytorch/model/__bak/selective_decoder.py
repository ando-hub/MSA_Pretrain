import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from mytorch.model.attentive_pooling import AttentivePooling


class SelectiveDecoder(nn.Module):
    def __init__(self, in_dim, dec_dim, dec_layer, out_dim, nhead=8, dropout_rate=0.1, pooling='gate'):
        super(MultiModalDecoder, self).__init__()
        self.in_dim = in_dim
        
        if pooling == 'ap':
            self.attn = AttentivePooling(in_dim, in_dim, nhead)
            attn_dim = in_dim * nhead
        elif pooling == 'gate':
            self.attn = GatePooling(in_dim, 3)  # num. modal = 3 (video, audio, text)
            attn_dim = in_dim
        else:
            raise ValueError('pooling must be either ap or gate')

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

    def forward(self, h_v, h_a, h_t):
        """
        Args:
            h_v (torch.Tensor): input video embedding tensor (B, embed_dim) or None 
            h_a (torch.Tensor): input audio embedding tensor (B, embed_dim) or None
            h_t (torch.Tensor): input text embedding tensor (B, embed_dim) or None

        Returns:
            (torch.Tensor): output embedding (B, out_dim)
        """

        nbat = [h.shape[0] for h in [h_v, h_a, h_t] if h is not None][0]
        device = [h.device for h in [h_v, h_a, h_t] if h is not None][0]
        h_v = h_v if h_v is not None else torch.zeros((nbat, self.in_dim)).float().to(device)
        h_a = h_a if h_a is not None else torch.zeros((nbat, self.in_dim)).float().to(device)
        h_t = h_t if h_t is not None else torch.zeros((nbat, self.in_dim)).float().to(device)
        
        x, att = self.attn(x, torch.ones(x.shape[0], device=x.device)*3, need_weights=True)
        x = self.dec(x)
        return x, att
