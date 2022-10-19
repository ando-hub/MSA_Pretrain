import torch
import torch.nn as nn
import torch.nn.functional as F
from mytorch.model.attentive_pooling import AttentivePooling
from mytorch.model.global_average_pooling import GlobalAveragePooling


class UniModalEncoder(nn.Module):
    def __init__(self, in_dim, enc_dim, enc_layer, dec_dim, dec_layer, out_dim,
                 nhead=8, dropout_rate=0.1, feat_dropout_rate=0.0, pooling='sap',
                 input_norm=False, in_layer=1):
        super(UniModalEncoder, self).__init__()
        # input feature processing
        self.input_norm = input_norm
        self.input_norm_layer = nn.LayerNorm(in_dim)
        self.input_layer_weights = nn.Parameter(torch.zeros(in_layer))

        # feature dropout
        self.feat_dropout = nn.Dropout(feat_dropout_rate)

        # modal encoder
        self.enc, enc_out_dim = self.__make_fc_layers(in_dim, enc_dim, enc_dim, enc_layer,
                                                      dropout_rate)
        # modal pooling
        self.pooling = pooling
        if pooling == 'sap':
            self.attn = AttentivePooling(enc_out_dim, enc_out_dim, nhead)
            attn_dim = enc_out_dim * nhead
        elif pooling == 'ap':
            self.attn = GlobalAveragePooling()
            attn_dim = enc_out_dim
        else:
            raise ValueError('pooling must be either sap or ap')

        # modal decoder
        self.dec, dec_out_dim = self.__make_fc_layers(attn_dim, out_dim, dec_dim, dec_layer,
                                                      dropout_rate, rm_final_activation=True)

    def __make_fc_layers(self, in_dim, out_dim, hid_dim, n_layer,
                         dropout_rate, rm_final_activation=False):
        layers = []
        prev_out_dim = in_dim
        for i in range(n_layer):
            current_out_dim = out_dim if i == n_layer-1 else hid_dim
            layers.append(nn.Linear(prev_out_dim, current_out_dim))
            if i != n_layer-1 or not rm_final_activation:
                layers.append(nn.LayerNorm(current_out_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            prev_out_dim = current_out_dim

        # convert nn.Sequential() instance
        seq = nn.Sequential(*layers) if layers else None
        return seq, prev_out_dim

    def forward(self, x, x_len):
        """
        Args:
            x (torch.Tensor): input tensor (B, T, in_dim) / (B, L, T, in_dim)
            x_len (torch.Tensor): input tensor length (B)

        Returns:
            (torch.Tensor): output embedding (B, N, emb_dim)
        """
        if x is None:
            return None, None
        else:
            x = self.feat_dropout(x)
            # normalize input featurs
            if self.input_norm:
                x = self.input_norm_layer(x)
            # weighted sum of multiple input layers
            if len(x.shape) == 4:
                nbat, nlay, nlen, ndim = x.shape
                norm_weights = F.softmax(self.input_layer_weights, dim=-1)
                norm_weights = norm_weights.unsqueeze(0).expand(nbat, nlay).unsqueeze(-1).unsqueeze(-1)
                x = (norm_weights * x).sum(axis=1)
            if self.enc:
                x = self.enc(x)
            if self.pooling == 'sap':
                x, att = self.attn(x, x_len, need_weights=True)
            elif self.pooling == 'ap':
                x = self.attn(x, x_len)
                att = None
            if self.dec:
                x = self.dec(x)
            return x, att
