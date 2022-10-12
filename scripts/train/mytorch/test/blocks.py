import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from pdb import set_trace
import numpy as np


class CNNBlock(nn.Module):
    def __init__(self, nin, nin_ch, cnn_ch, cnn_ker, cnn_str, pool, pool_ker,
                 batchnorm=False, activ=False):
        super(CNNBlock, self).__init__()
        _layers = []
        # CNN
        if cnn_ch:
            _pad = tuple([int(np.floor(v/2.)) for v in cnn_ker])
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

        # Batchnorm
        if batchnorm:
            _layer = nn.BatchNorm2d(cnn_ch)
            _layers.append(_layer)

        # Activation
        if activ:
            if activ == 'relu':
                _layer = nn.ReLU()
            elif activ == 'tanh':
                _layer = nn.Tanh()
            else:
                raise NotImplementedError('invalid activation: {}'.format(activ))
            _layers.append(_layer)

        # Pooling
        if pool or pool.lower() != 'none':
            _pad = (0, 0)
            _str = pool_ker
            if pool == 'max':
                _layer = nn.MaxPool2d(pool_ker, padding=_pad, stride=_str)
            elif pool == 'avg':
                _layer = nn.AvgPool2d(pool_ker, padding=_pad, stride=_str)
            else:
                raise NotImplementedError('invalid pooling: {}'.format(pool))
            _layers.append(_layer)
            self.pool_ker = pool_ker
            self.pool_pad = _pad
            self.pool_str = _str
            self.pool_dil = (1, 1)

        # set layers
        self.layers = nn.Sequential(*_layers)
        self.outch = cnn_ch
        self.nin = torch.tensor(nin)
        return

    def __mask(self, x, x_len):
        # length convert by conv2d
        x_len = self.__calclen(
                x_len,
                self.cnn_pad[0], self.cnn_dil[0], self.cnn_ker[0], self.cnn_str[0]
                )
        # length convert by pooling
        x_len = self.__calclen(
                x_len,
                self.pool_pad[0], self.pool_dil[0], self.pool_ker[0], self.pool_str[0]
                )

        # zero masking
        mask = torch.zeros(x.shape).bool().to(x.device)
        for i, j in enumerate(x_len):
            mask[i, :, j:, :] = True
        x = x.masked_fill_(mask, 0.)
        return x, x_len

    def __calclen(self, xin, p, d, k, s):
        return torch.floor((xin.float() + 2*p - d*(k-1) - 1)/s + 1).long()

    def forward(self, x, x_len):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)       # reshape: [nbat x 1 x nlen x ndim]
        x = self.layers(x)
        x, x_len = self.__mask(x, x_len)
        return x, x_len

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


class LSTMBlock(nn.Module):
    def __init__(self, nin, nhid, nlay, bidirec=False, drop=0.0):
        super(LSTMBlock, self).__init__()
        layers = []

        # LSTM
        _dropout = 0.0 if nlay == 1 else drop
        _layer = nn.LSTM(
                nin, nhid, nlay,
                batch_first=True,
                bidirectional=bidirec,
                dropout=_dropout,
                )
        layers.append(_layer)
        self.nhid = nhid
        self.nlay = nlay
        self.n_direc = 2 if bidirec else 1
        self.nout = nhid * self.n_direc

        # dropout
        if drop > 0:
            layers.append(nn.Dropout(drop))

        # set layers
        self.layers = nn.Sequential(*layers)

    def init_state(self, x):
        batch_size = x.shape[0]
        device = x.device
        # init values
        h0 = torch.zeros(
                (self.n_direc*self.nlay, batch_size, self.nhid)
                ).to(device)
        c0 = torch.zeros(
                (self.n_direc*self.nlay, batch_size, self.nhid)
                ).to(device)
        return h0, c0

    def get_outdim(self):
        return self.nout

    def forward(self, x, x_len):
        h0, c0 = self.init_state(x)
        for l in self.layers:
            if isinstance(l, nn.LSTM):
                x = pack_padded_sequence(x, x_len, batch_first=True)
                x, _ = l(x, (h0, c0))
                x, x_len = pad_packed_sequence(x, batch_first=True)
            else:
                x = l(x)
        return x, x_len


class FlattenBlock(nn.Module):
    def __init__(self, nin, nin_ch):
        super(FlattenBlock, self).__init__()
        self.nout = nin*nin_ch

    def forward(self, x, x_len):
        if len(x.shape) != 4:
            raise ValueError('Input must be 4-dim tensor!')
        nbat, nch, nlen, ndim = x.shape
        x = x.transpose(1, 2)                   # [nbat x nlen x nch x ndim]
        x = x.contiguous().view(nbat, nlen, -1) # [nbat x nlen x (nch*ndim)]
        return x, x_len

    def get_outdim(self):
        return self.nout

"""
class StructuredSelfAttention(nn.Module):
    def __init__(self, nin, nhid=0, nhead=1, drop=0.0):
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

    def get_outdim(self):
        return self.nout

    def forward(self, x, x_len):
        nbat, nlen, ndim = x.size()

        # prepare attention mask for zero-padding
        mask = torch.arange(nlen)[None, :] >= x_len[:, None]
        #mask = mask.unsqueeze(-1).expand_as(x).to(x.device)
        mask = mask.unsqueeze(-1).repeat(1,1,self.nhead).to(x.device)

        # attention
        xres = x.contiguous().view(-1, ndim)        # batch*sourceL x ndim
        attn_ = self.layers(xres)                   # batch*sourceL x nhead
        attn_ = attn_.view(nbat, -1, self.nhead)    # batch x sourceL x nhead
        attn_ = attn_.masked_fill_(mask, -float('inf')) # overwrite  -inf
        attn = F.softmax(attn_, dim=1)              # batch x sourceL x nhead
        y_ = torch.matmul(attn.transpose(1, 2), x)  # batch x nhead x Ndim
        y = torch.sum(y_, dim=1)/self.nhead         # batch x Ndim (averaging)
        return y, attn
"""
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
        attn = self.layers(x.contiguous().view(-1, ndim))   # batch*sourceL x nhead
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



class MultiheadAttentionBlock(nn.Module):
    """
    Wrapped Block of torch.nn.MultiheadAttention
    """
    def __init__(self, nin, nhead=1, drop=0.0):
        super(MultiheadAttentionBlock, self).__init__()
        l = nn.MultiheadAttention(nin, nhead, dropout=drop)
        self.layers = nn.Sequential(l)
        self.nout = nin

    def get_outdim(self):
        return self.nout

    def _get_mask(self, x_len):
        return torch.arange(x_len.max().item())[None, :] >= x_len[:, None]

    def forward(self, x, x_len):
        # prepare mask
        mask = self._get_mask(x_len).to(x.device)

        # decode
        x = x.transpose(0, 1)  # [nlen, nbat, ndim]
        x, _ = self.layers[0](
                x, x, x,
                key_padding_mask=mask,
                need_weights=False
                )
        x = x[0]        # slice head indices
        x_len = torch.ones(x_len.shape, dtype=torch.int64, requires_grad=False)
        del mask
        return x, x_len


class LinearBlock(nn.Module):
    def __init__(self, nin, nhid, nlay=1, naux=0, activ=None, drop=0.0, batchnorm=False, linear_type='linear'):
        super(LinearBlock, self).__init__()
        # error check
        if linear_type not in ['linear', 'aux', 'auxadd', 'switch']:
            raise ValueError('invalid linear_type: {}'.format(linear_type))
        elif linear_type in ['aux', 'auxadd', 'switch'] and naux <= 0:
            raise ValueError('naux > 0 if linear_type is aux or switch')

        layers = []
        for i in range(nlay):
            _nhid = nin if i == 0 else nhid
            if linear_type == 'linear':
                l = nn.Linear(_nhid, nhid)
            elif linear_type == 'aux':
                l = nn.Linear(_nhid+naux, nhid)
            elif linear_type == 'auxadd':
                l = nn.Linear(_nhid, nhid)
            elif linear_type == 'switch':
                _l = {str(j): nn.Linear(_nhid, nhid) for j in range(naux)}
                l = nn.ModuleDict(_l)
            layers.append(l)

            if batchnorm:
                layers.append(nn.BatchNorm1d(nhid))

            if activ:
                if activ == 'relu':
                    layers.append(nn.ReLU())
                elif activ == 'tanh':
                    layers.append(nn.Tanh())
                elif activ == 'softmax':
                    layers.append(nn.Softmax(dim=1))
                else:
                    raise ValueError('invalid activation: {}'.format(activ))

            if drop:
                layers.append(nn.Dropout(drop))

        self.layers = nn.Sequential(*layers)
        self.naux = naux
        self.nout = nhid
        self.linear_type = linear_type
        if linear_type == 'auxadd':
            self.add_layer = nn.Linear(naux, nhid)

    def get_outdim(self):
        return self.nout

    def forward(self, x, x_len=None, aux=None):
        for l in self.layers:
            if isinstance(l, nn.ModuleDict):
                nbat, ndim = x.shape    # 2-d input only
                _x = torch.zeros(
                        (nbat, self.nout),
                        dtype=torch.float32,
                        device=x.device)
                # weighted sum
                for i in range(self.naux):
                    _x += (l[str(i)](x).T * aux[:, i]).T
                x = _x
            elif isinstance(l, nn.Linear) and self.linear_type == 'aux':
                x = l(torch.cat((x, aux), 1))
            elif isinstance(l, nn.Linear) and self.linear_type == 'auxadd':
                x = l(x) + self.add_layer(aux)
            elif isinstance(l, nn.BatchNorm1d) and len(x.shape) == 3:
                nbat, nlen, ndim = x.shape
                x = l(x.view(-1, ndim)).view(nbat, nlen, ndim)
            else:
                x = l(x)
        return x, x_len


class AdaptiveLinearBlock(nn.Module):
    def __init__(self, nin, nhid, nlay=1, naux=0, activ=None, drop=0.0, batchnorm=False):
        super(AdaptiveLinearBlock, self).__init__()
        # error check
        if naux <= 0:
            raise ValueError('naux > 0 is required')

        layers = []
        for i in range(nlay):
            _nhid = nin if i == 0 else nhid
            l = nn.Linear(_nhid, nhid)
            layers.append(l)

            if batchnorm:
                layers.append(nn.BatchNorm1d(nhid))

            if activ:
                if activ == 'relu':
                    layers.append(nn.ReLU())
                elif activ == 'tanh':
                    layers.append(nn.Tanh())
                elif activ == 'softmax':
                    layers.append(nn.Softmax(dim=1))
                else:
                    raise ValueError('invalid activation: {}'.format(activ))

            if drop:
                layers.append(nn.Dropout(drop))

        self.layers = nn.Sequential(*layers)
        self.naux = naux
        self.nout = nhid

        self.add_layer = nn.Linear(naux, nhid, bias=False)

    def get_outdim(self):
        return self.nout

    def forward(self, x, x_len=None, aux=None):
        for l in self.layers:
            if isinstance(l, nn.Linear):
                x = l(x) + self.add_layer(aux)
            elif isinstance(l, nn.BatchNorm1d) and len(x.shape) == 3:
                nbat, nlen, ndim = x.shape
                x = l(x.view(-1, ndim)).view(nbat, nlen, ndim)
            else:
                x = l(x)
        return x, x_len


class AdaptiveCNNBlock(nn.Module):
    def __init__(self, nin, nin_ch, cnn_ch, cnn_ker, cnn_str, pool, pool_ker, naux=0,
                 batchnorm=False, activ=False):
        super(AdaptiveCNNBlock, self).__init__()
        # error check
        if naux <= 0:
            raise ValueError('naux > 0 is required')

        _layers = []
        # CNN
        if cnn_ch:
            _pad = tuple([int(np.floor(v/2.)) for v in cnn_ker])
            if isinstance(cnn_str, int):
                _str = (cnn_str, cnn_str)
            else:
                _str = tuple(cnn_str)

            """
            _layer = nn.Conv2d(
                    nin_ch, cnn_ch, cnn_ker, padding=_pad, stride=_str
                    )
            _layers.append(_layer)
            """
            self.cnn_layer = F.conv2d
            self.cnn_filter_generator = nn.Linear(naux, nin_ch*cnn_ch*np.prod(cnn_ker))
            self.cnn_filter_bias_generator = nn.Linear(naux, cnn_ch)

            self.cnn_ker = cnn_ker
            self.cnn_pad = _pad
            self.cnn_str = _str
            self.cnn_dil = (1, 1)

        # Batchnorm
        if batchnorm:
            _layer = nn.BatchNorm2d(cnn_ch)
            _layers.append(_layer)

        # Activation
        if activ:
            if activ == 'relu':
                _layer = nn.ReLU()
            elif activ == 'tanh':
                _layer = nn.Tanh()
            else:
                raise NotImplementedError('invalid activation: {}'.format(activ))
            _layers.append(_layer)

        # Pooling
        if pool or pool.lower() != 'none':
            _pad = (0, 0)
            _str = pool_ker
            if pool == 'max':
                _layer = nn.MaxPool2d(pool_ker, padding=_pad, stride=_str)
            elif pool == 'avg':
                _layer = nn.AvgPool2d(pool_ker, padding=_pad, stride=_str)
            else:
                raise NotImplementedError('invalid pooling: {}'.format(pool))
            _layers.append(_layer)
            self.pool_ker = pool_ker
            self.pool_pad = _pad
            self.pool_str = _str
            self.pool_dil = (1, 1)

        # set layers
        self.layers = nn.Sequential(*_layers)
        self.outch = cnn_ch
        self.inch = nin_ch
        self.nin = torch.tensor(nin)
        return

    def __gen_filter(self, aux):
        filt = self.cnn_filter_generator(aux)
        filt = filt.view(aux.shape[0], self.outch, self.inch, *self.cnn_ker)
        bias = self.cnn_filter_bias_generator(aux)
        return filt, bias

    def __mask(self, x, x_len):
        # length convert by conv2d
        x_len = self.__calclen(
                x_len,
                self.cnn_pad[0], self.cnn_dil[0], self.cnn_ker[0], self.cnn_str[0]
                )
        # length convert by pooling
        x_len = self.__calclen(
                x_len,
                self.pool_pad[0], self.pool_dil[0], self.pool_ker[0], self.pool_str[0]
                )

        # zero masking
        mask = torch.zeros(x.shape).bool().to(x.device)
        for i, j in enumerate(x_len):
            mask[i, :, j:, :] = True
        x = x.masked_fill_(mask, 0.)
        return x, x_len

    def __calclen(self, xin, p, d, k, s):
        return torch.floor((xin.float() + 2*p - d*(k-1) - 1)/s + 1).long()

    def adaptive_convolution(self, x, aux):
        filt, bias = self.__gen_filter(aux)
        x = torch.cat([F.conv2d(xi.unsqueeze(0), fi, bi, stride=self.cnn_str, padding=self.cnn_pad)
                for xi, fi, bi in zip(x, filt, bias)],
                0)
        return x

    def forward(self, x, x_len, aux):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)       # reshape: [nbat x 1 x nlen x ndim]
        # x = self.layers(x)
        #filt, bias = self.__gen_filter(aux)
        #set_trace()
        #x = self.cnn_layer(x, filt, bias=bias, padding=self.cnn_pad, stride=self.cnn_str)
        x = self.adaptive_convolution(x, aux)
        x = self.layers(x)
        x, x_len = self.__mask(x, x_len)
        return x, x_len

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


class AdaptiveLSTMBlock(nn.Module):
    def __init__(self, nin, nhid, nlay, naux=0, bidirec=False, drop=0.0):
        super(AdaptiveLSTMBlock, self).__init__()
        # error check
        if naux <= 0:
            raise ValueError('naux > 0 is required')

        #layers = []
        # linear for adaptation
        self.adapt_layer = nn.Linear(naux, nin)

        # LSTM
        _dropout = 0.0 if nlay == 1 else drop
        """
        _layer = nn.LSTM(
                nin, nhid, nlay,
                batch_first=True,
                bidirectional=bidirec,
                dropout=_dropout,
                )
        layers.append(_layer)
        """
        self.lstm_layer = nn.LSTM(
                nin, nhid, nlay,
                batch_first=True,
                bidirectional=bidirec,
                dropout=_dropout,
                )
        self.nhid = nhid
        self.nlay = nlay
        self.n_direc = 2 if bidirec else 1
        self.nout = nhid * self.n_direc

        # dropout
        if drop > 0:
            # layers.append(nn.Dropout(drop))
            self.dropout_layer = nn.Dropout(drop)

        self.drop = drop

        # set layers
        # self.layers = nn.Sequential(*layers)

    def init_state(self, x):
        batch_size = x.shape[0]
        device = x.device
        # init values
        h0 = torch.zeros(
                (self.n_direc*self.nlay, batch_size, self.nhid)
                ).to(device)
        c0 = torch.zeros(
                (self.n_direc*self.nlay, batch_size, self.nhid)
                ).to(device)
        return h0, c0

    def get_outdim(self):
        return self.nout

    def forward(self, x, x_len, aux):
        h0, c0 = self.init_state(x)
        aux = self.adapt_layer(aux)
        x = x + aux.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = pack_padded_sequence(x, x_len, batch_first=True)
        x, _ = self.lstm_layer(x, (h0, c0))
        x, x_len = pad_packed_sequence(x, batch_first=True)
        if self.drop > 0.0:
            x = self.dropout_layer(x)
        return x, x_len


class NoProcBlock(nn.Module):
    def __init__(self, nin):
        super(NoProcBlock, self).__init__()
        self.nin = nin

    def forward(self, x, x_len=None):
        return x, x_len

    def get_outdim(self):
        return self.nin


class TDNNBlock(nn.Module):
    def __init__(self, nin, nhid, context, batchnorm=False, activ=None, drop=0.0, stride=1):
        '''
        https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            kernel size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            kernel size 3 and dilation 2 is equivalent to [-2, 0, 2]
            kernel size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNNBlock, self).__init__()
        self.kernel_size, self.dilation = self._get_context_params(context)
        self.stride = stride
        #self.padding = self.dilation * (self.kernel_size - 1) // 2
        self.padding = 0
        self.input_dim = nin
        self.output_dim = nhid

        # CNN -> (batchnorm) -> (activation) -> (dropout)
        layers = []
        l = nn.Conv1d(self.input_dim, self.output_dim, self.kernel_size,
                      stride=self.stride,
                      padding=self.padding,
                      dilation=self.dilation)
        layers.append(l)
        if batchnorm:
            layers.append(nn.BatchNorm1d(nhid))
        if activ:
            if activ == 'relu':
                layers.append(nn.ReLU())
            elif activ == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError('invalid activation: {}'.format(activ))
        if drop:
            layers.append(nn.Dropout(drop))

        self.layers = nn.Sequential(*layers)
        self.nout = nhid

    def _get_context_params(self, context):
        context_size = len(context)
        dilation = 1 if context_size == 1 else context[1]-context[0]

        _context = [context[0] + i*dilation for i in range(context_size)]
        assert context == _context, '{} is not size {}, dilat {}'.format(
                context, context_size, dilation)
        return context_size, dilation

    def get_outdim(self):
        return self.output_dim

    def output_lengths(self, in_lengths):
        out_lengths = (
            in_lengths + 2*self.padding - self.dilation*(self.kernel_size-1) +
            self.stride - 1
        ) // self.stride
        return out_lengths

    def forward(self, x, x_len=None):
        '''
        input: size (Batch, T_len, Dim)
        outpu: size (Batch, T_len_new, Dim_new)
        '''
        assert len(x.size()) == 3           # x is of size (B, T, D)
        x = x.transpose(1, 2).contiguous()  # turn x to (B, D, T) for tdnn
        x = self.layers(x)
        x = x.transpose(2, 1).contiguous()  # turn it back to (B, T, D)
        x_len = self.output_lengths(x_len)
        return x, x_len


class TransformerEncoderBlock(nn.Module):
    def __init__(self, nin, nhid, nlay, nhead, activ='relu', drop=0.0):
        super(TransformerEncoderBlock, self).__init__()
        enc = nn.TransformerEncoderLayer(
                d_model=nin, nhead=nhead,
                dim_feedforward=nhid,
                dropout=drop,
                activation=activ,
                )
        layers = [nn.TransformerEncoder(enc, num_layers=nlay)]
        self.layers = nn.Sequential(*layers)
        self.nout = nin

    def get_outdim(self):
        return self.nout

    def get_mask(self, x_len):
        max_len = x_len.max().item()
        return torch.arange(max_len)[None, :] >= x_len[:, None]

    def forward(self, x, x_len):
        # reshape [nbat, nlen, ndim] -> [nlen, nbat, ndim]
        x = x.transpose(0, 1)
        x = self.layers[0](
                x, src_key_padding_mask=self.get_mask(x_len).to(x.device)
                )

        x = x[0]        # slice head indices
        x_len = torch.ones(x_len.shape, dtype=torch.int64)
        return x, x_len

