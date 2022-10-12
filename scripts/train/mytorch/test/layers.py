# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math
from pdb import set_trace


class MyCNN(nn.Module):
    def __init__(self, nin, nin_ch, cnn_ch, cnn_ker, cnn_str, pool, pool_ker,
                 batchnorm=False, activ=False):
        super(MyCNN, self).__init__()
        _layers = []
        if cnn_ch:
            _pad = tuple([int(math.floor(v/2.)) for v in cnn_ker])
            if isinstance(cnn_str, int):
                _str = (cnn_str, cnn_str)
            else:
                _str = tuple(cnn_str)

            _layer = nn.Conv2d(
                    nin_ch, cnn_ch, cnn_ker, padding=_pad, stride=_str
                    )
            _layers.append(_layer)
            self.cnn_ker = cnn_ker
            self.cnn_pad = _pad
            self.cnn_str = _str
            self.cnn_dil = (1, 1)

        if batchnorm:
            _layer = nn.BatchNorm2d(cnn_ch)
            _layers.append(_layer)

        if activ:
            if activ == 'relu':
                _layer = nn.ReLU()
            elif activ == 'tanh':
                _layer = nn.Tanh()
            else:
                raise NotImplementedError('No exist activation: {}'.format(activ))
            _layers.append(_layer)

        if pool or pool.lower() != 'none':
            _pad = (0, 0)
            _str = pool_ker
            if pool == 'max':
                _layer = nn.MaxPool2d(pool_ker, padding=_pad, stride=_str)
            elif pool == 'avg':
                _layer = nn.AvgPool2d(pool_ker, padding=_pad, stride=_str)
            else:
                raise NotImplementedError('No exist pooling: {}'.format(pool))
            _layers.append(_layer)
            self.pool_ker = pool_ker
            self.pool_pad = _pad
            self.pool_str = _str
            self.pool_dil = (1, 1)
        self.layers = nn.Sequential(*_layers)
        self.outch = cnn_ch
        self.nin = torch.tensor(nin)
        return

    def __mask(self, y, x_len):
        # length convert by conv2d
        h_len = self.__calclen(
                x_len, self.cnn_pad[0], self.cnn_dil[0], self.cnn_ker[0], self.cnn_str[0]
                )
        # length convert by pooling
        y_len = self.__calclen(
                h_len, self.pool_pad[0], self.pool_dil[0], self.pool_ker[0], self.pool_str[0]
                )

        mask = torch.zeros(y.shape).bool().to(y.device)
        for i, j in enumerate(y_len):
            mask[i, :, j:, :] = True
        y_masked = y.masked_fill_(mask, 0.)
        return y_masked, y_len

    def __calclen(self, xin, p, d, k, s):
        return torch.floor((xin.float() + 2*p - d*(k-1) - 1)/s + 1).long()

    def forward(self, x, x_len):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)       # reshape: [nbat x 1 x nlen x ndim]
        y = self.layers(x)
        y, y_len = self.__mask(y, x_len)
        return y, y_len

    def get_outch(self):
        return self.outch

    def get_outdim(self):
        _ndim = self.__calclen(
                self.nin, self.cnn_pad[1], self.cnn_dil[1], self.cnn_ker[1], self.cnn_str[1]
                )
        _ndim = self.__calclen(
                _ndim, self.pool_pad[1], self.pool_dil[1], self.pool_ker[1], self.pool_str[1]
                )
        return int(_ndim)


class MyLSTM(nn.Module):
    def __init__(self, nin, nhid, nlay, bidirec=False, drop=0.0, batchnorm=False):
        super(MyLSTM, self).__init__()
        self.layer = nn.LSTM(
                nin, nhid, nlay,
                batch_first=True,
                bidirectional=bidirec,
                dropout=drop,
                )

        self.nhid = nhid
        self.nlay = nlay
        self.n_direc = 2 if bidirec else 1
        self.nout = nhid * self.n_direc
        if drop > 0:
            self.is_dropout = True
            self.dropout = nn.Dropout(drop)
        else:
            self.is_dropout = False

    def init_state(self, x):
        batch_size = x.shape[0]
        device = x.device

        h0 = torch.zeros((self.n_direc*self.nlay, batch_size, self.nhid)).to(device)
        c0 = torch.zeros((self.n_direc*self.nlay, batch_size, self.nhid)).to(device)
        return h0, c0

    def get_outdim(self):
        return self.nout

    def forward(self, x, x_len):
        h0, c0 = self.init_state(x)
        x_pack = pack_padded_sequence(x, x_len, batch_first=True)
        y_pack, (h, c) = self.layer(x_pack, (h0, c0))
        y, y_len = pad_packed_sequence(y_pack, batch_first=True)
        if self.is_dropout:
            y = self.dropout(y)
        return y, y_len

    def sort_forward(self, x, x_len):
        sorted_x_len, sorted_indices = torch.sort(x_len, dim=0, descending=True)
        sorted_x = x[sorted_indices]
        y = self.forward(sorted_x, sorted_x_len)
        # re-sort
        origin_indices = torch.zeros(x_len.shape).long()
        for i, v in enumerate(sorted_indices):
            origin_indices[v] = i
        orig_y = y[origin_indices]
        return orig_y, x_len


class MyTransformer(nn.Module):
    def __init__(self, nin, nhead, nlay_enc, nlay_dec, nhid, drop=0.0, activation='relu'):
        self.layer = nn.Transformer(nin, nhead, nlay_enc, nlay_dec, nhid, drop, activation)

    def forward(self, x, x_len):
        mask = None
        binmask = None
        y = self.layer(
                x, x,
                src_mask=mask, tgt_mask=mask, memory_mask=mask,
                src_key_padding_mask=binmask,
                tgt_key_padding_mask=binmask,
                memory_key_padding_mask=binmask
                )
        return y, x_len


class MyFlatten(nn.Module):
    def __init__(self, nin, nin_ch):
        super(MyFlatten, self).__init__()
        self.nout = nin*nin_ch

    def forward(self, x, x_len):
        if len(x.shape) != 4:
            raise ValueError('Input must be 4-dim tensor!')
        nbat, nch, nlen, ndim = x.shape             # store CNN-output tensor shape
        y = x.transpose(1, 2)                       # reshape: [nbat x nlen x nch x ndim]
        y = y.contiguous().view(nbat, nlen, -1)     # reshape: [nbat x nlen x (nch*ndim)]
        return y, x_len

    def get_outdim(self):
        return self.nout


class StructuredSelfAttention(nn.Module):
    """
    [ref] https://github.com/kaushalshetty/Structured-Self-Attention/blob/master/attention/model.py
    """
    def __init__(self, nin, nhid=0, drop=0.0, nhead=1, L2weight=None, average=True):
        super(StructuredSelfAttention, self).__init__()
        if nhid <= 0:
            nhid = nin

        self.layers = nn.Sequential(
                nn.Linear(nin, nhid),
                nn.Tanh(),
                nn.Dropout(drop),
                nn.Linear(nhid, nhead)
                )
        self.nhead = nhead
        self.nout = nin
        self.average = average

    def get_outdim(self):
        if self.average:
            return self.nout
        else:
            return self.nhead * self.nout

    def forward(self, x, x_len):
        nbat, nlen, ndim = x.size()

        # prepare attention mask for zero-padding
        mask = torch.arange(nlen)[None, :] >= x_len[:, None]
        mask = mask.unsqueeze(-1).repeat(1,1,self.nhead).to(x.device)

        # attention
        x = x.contiguous().view(-1, ndim)               # batch*sourceL x ndim
        attn = self.layers(x)                           # batch*sourceL x nhead
        attn = attn.view(nbat, -1, self.nhead)          # batch x sourceL x nhead
        # attn_ = attn_ + mask                          # overwrite padded parts as -inf
        attn = attn.masked_fill_(mask, -float('inf'))   # overwrite padded parts as -inf
        attn = F.softmax(attn, dim=1)                   # batch x sourceL x nhead
        x = torch.matmul(attn.transpose(1, 2), x)       # batch x nhead x Ndim
        if self.average:
            x = torch.sum(x, dim=1)/self.nhead          # batch x Ndim (averaging)
        else:
            x = x.view(nbat, -1)                        # batch x Ndim (averaging)
        return x, attn


class MyMultiheadAttention(nn.Module):
    """
    Wrapped of torch.nn.MultiheadAttention
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MyMultiheadAttention, self).__init__()
        self.layer = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x, x_len):
        input_mask = None
        attn_mask = None
        y, attn = self.layer(
                x, x, x,
                key_padding_mask=input_mask,
                attn_mask=attn_mask,
                need_weights=True
                )
        return y, attn


class MyLinear(nn.Module):
    def __init__(self, nin, nhid, nlay=1, activ='relu', drop=0.0, batchnorm=False):
        super(MyLinear, self).__init__()
        layers = []
        _nhid = nin
        for i in range(nlay):
            _nhid = nin if i == 0 else nhid
            l = nn.Linear(_nhid, nhid)
            layers.append(l)
            if activ:
                if activ == 'relu':
                    layers.append(nn.ReLU())
                elif activ == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    raise NotImplementedError('No exist activation: {}'.format(activ))
            if drop:
                layers.append(nn.Dropout(drop))
            if batchnorm:
                layers.append(nn.BatchNorm1d(nhid))

        self.layers = nn.Sequential(*layers)
        self.nout = nhid

    def get_outdim(self):
        return self.nout

    def forward(self, x, x_len=None):
        y = self.layers(x)
        return y, x_len


class MyLinearCond(nn.Module):
    def __init__(self, nin, ncond, nhid, activ='relu', drop=0.0, batchnorm=False):
        super(MyLinearCond, self).__init__()
        if ncond <= 0:
            raise ValueError('ncond must be > 0 !')

        layers = []
        layers.append(nn.Linear(nin+ncond, nhid))
        if activ:
            if activ == 'relu':
                layers.append(nn.ReLU())
            elif activ == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise NotImplementedError('No valid activation: {}'.format(activ))
        if drop:
            layers.append(nn.Dropout(drop))
        if batchnorm:
            layers.append(nn.BatchNorm1d(nhid))
        self.layers = nn.Sequential(*layers)
        self.nout = nhid

    def get_outdim(self):
        return self.nout

    def forward(self, x, x_len=None, cond=None):
        y = self.layers(torch.cat((x, cond), 1))
        return y, x_len


class MyLinearSlct(nn.Module):
    def __init__(self, nin, nslct, nhid, activ='relu', drop=0.0, batchnorm=False):
        super(MyLinearSlct, self).__init__()
        if nslct <= 0:
            raise ValueError('nslct must be > 0 !')

        layerdict = {str(j): nn.Linear(nin, nhid) for j in range(nslct)}
        self.layerdict = nn.ModuleDict(layerdict)

        layers = []
        if activ:
            if activ == 'relu':
                layers.append(nn.ReLU())
            elif activ == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise NotImplementedError('No valid activation: {}'.format(activ))
        if drop:
            layers.append(nn.Dropout(drop))
        if batchnorm:
            layers.append(nn.BatchNorm1d(nhid))
        self.layers = nn.Sequential(*layers)
        self.nout = nhid

    def get_outdim(self):
        return self.nout

    def forward(self, x, x_len=None, slct=None):
        y = torch.zeros((x.shape[0], self.nout), dtype=torch.float32).to(x.device)
        for i, j in enumerate(slct.argmax(1)):
            y[i] = self.layerdict[str(int(j))](x[i])
        y = self.layers(y)
        return y, x_len



"""
## for debug

def _print(head, var, dim=1, gpu=False):
    v = var.data.cpu() if gpu else var.data
    print head
    #print v.numpy().tolist()
    if dim == 1: __print1d(v)
    elif dim == 2: __print2d(v)
    elif dim == 3: __print3d(v)
    else: raise ValueError

def _print_packed_sequence(head, var, dim=1, gpu=False):
    v = var.data.data.cpu() if gpu else var.data.data
    print head
    if dim == 1: __print1d(v)
    elif dim == 2:  __print2d(v)
    else: raise ValueError

def __print1d(v):
    print ' '.join(['{:.13f}'.format(vi) for vi in v.numpy().flatten().tolist()])

def __print2d(v):
    for l in v.numpy().tolist():
        print ' '.join(['{:.13f}'.format(vi) for vi in l])

def __print3d(v):
    for l in v.numpy().tolist():
        for l2 in l:
            print ' '.join(['{:.13f}'.format(vi) for vi in l2])

"""
