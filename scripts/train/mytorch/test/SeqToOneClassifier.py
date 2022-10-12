import torch.nn as nn
import copy

from .blocks import (
        CNNBlock, LSTMBlock, LinearBlock, FlattenBlock,
        StructuredSelfAttention, MultiheadAttentionBlock,
        TDNNBlock, TransformerEncoderBlock,
        NoProcBlock,
        AdaptiveCNNBlock, AdaptiveLSTMBlock, AdaptiveLinearBlock,
                    )
from pdb import set_trace


class SeqToOne(nn.Module):
    def __init__(self, nin, nout, cfg, naux=0):
        super(SeqToOne, self).__init__()
        self.main_struct = cfg['model']['main_struct']
        self.aux_struct = cfg['model']['aux_struct']
        self.nout = nout

        # store encoder struct (input ~ 'att')
        self.main_struct_enc = []
        for l in self.main_struct:
            self.main_struct_enc.append(l)
            if l.startswith('att') or l.startswith('multiatt'):
                break

        # load aux layers
        if naux > 0:
            self.aux_layers, nauxout = self._load_blocks(
                    self.aux_struct, cfg['modelparam'], naux
                    )
        else:
            self.aux_layers = None
            nauxout = 0

        # load main layers
        self.main_layers, nmidout = self._load_blocks(
                self.main_struct, cfg['modelparam'], nin,
                nout=self.nout, naux=nauxout
                )

        # load (multi-) output layers
        _linear_type = cfg['model']['aux'] if cfg['model']['aux'] else 'linear'
        output_layers = []
        for _nout in nout:
            if _linear_type == 'alinear':
                _l = AdaptiveLinearBlock(
                        nmidout, _nout, nlay=1, naux=nauxout,
                        activ=None, drop=0., batchnorm=False,
                        )
            else:
                _l = LinearBlock(
                        nmidout, _nout, nlay=1, naux=nauxout,
                        activ=None, drop=0., batchnorm=False,
                        linear_type=_linear_type
                        )
            output_layers.append(_l)
        self.output_layers = nn.Sequential(*output_layers)

        # initialize parameters
        for name, param in self.named_parameters():
            param.data.normal_(0., 0.01)
        return

    def _load_blocks(self, struct, modelparam, nin, nout=0, naux=0):
        layerdict = {}
        nin_ch = 1
        for i, name in enumerate(struct):
            # prepare config
            if name in modelparam:
                _param = copy.deepcopy(modelparam[name])
            else:
                _param = None

            # add blocks
            if name.startswith('cnn'):
                _l = CNNBlock(nin, nin_ch, **_param)
                nin_ch = _l.get_outch()
            elif name.startswith('flat'):
                _l = FlattenBlock(nin, nin_ch)
            elif name.startswith('lstm'):
                _l = LSTMBlock(nin, **_param)
            elif name.startswith('att'):
                _l = StructuredSelfAttention(nin, **_param)
            elif name.startswith('multiatt'):
                _l = MultiheadAttentionBlock(nin, **_param)
            elif name.startswith('linear') or name.startswith('full'):
                if name.endswith('out') and nout > 0:
                    _nhid = nout
                    del _param['nhid']
                else:
                    _nhid = _param.pop('nhid')
                _l = LinearBlock(nin, _nhid, naux=naux, **_param)
            elif name.startswith('no'):
                _l = NoProcBlock(nin)
            elif name.startswith('tdnn'):
                _l = TDNNBlock(nin, **_param)
            elif name.startswith('trans'):
                _l = TransformerEncoderBlock(nin, **_param)
            elif name.startswith('acnn'):
                _l = AdaptiveCNNBlock(nin, nin_ch, naux=naux, **_param)
                nin_ch = _l.get_outch()
            elif name.startswith('alstm'):
                _l = AdaptiveLSTMBlock(nin, naux=naux, **_param)
            elif name.startswith('alinear'):
                if name.endswith('out') and nout > 0:
                    _nhid = nout
                    del _param['nhid']
                else:
                    _nhid = _param.pop('nhid')
                _l = AdaptiveLinearBlock(nin, _nhid, naux=naux, **_param)
            else:
                raise ValueError('invalid layer name: {}'.format(name))

            # set _l to layerdict
            layerdict[name] = _l
            nin = _l.get_outdim()

        return nn.ModuleDict(layerdict), nin


    def forward(self, x, x_len, aux=None):
        # aux decoding
        if aux is not None and self.aux_layers is not None:
            for name in self.aux_struct:
                aux, _ = self.aux_layers[name](aux)

        # main decoding
        for name in self.main_struct:
            _layer = self.main_layers[name]
            if ((isinstance(_layer, LinearBlock) and _layer.linear_type != 'linear')
                or isinstance(_layer, AdaptiveCNNBlock)
                or isinstance(_layer, AdaptiveLSTMBlock)
                or isinstance(_layer, AdaptiveLinearBlock)):
                x, x_len = _layer(x, x_len, aux)
            else:
                x, x_len = _layer(x, x_len)

        # output
        y = []
        for l in self.output_layers:
            if isinstance(l, AdaptiveLinearBlock) or l.linear_type != 'linear':
                y.append(l(x, x_len, aux)[0])
            else:
                y.append(l(x, x_len)[0])
        del x, x_len
        return y

    """
    def forward_enc(self, x, x_len, aux=None):
        for name in self.main_struct_enc:
            _layer = self.main_layers[name]
            if ((isinstance(_layer, LinearBlock) and _layer.linear_type != 'linear')
                or isinstance(_layer, AdaptiveCNNBlock)
                or isinstance(_layer, AdaptiveLSTMBlock)
                or isinstance(_layer, AdaptiveLinearBlock)):
                x, x_len = _layer(x, x_len, aux)
            else:
                x, x_len = _layer(x, x_len)
        return x, x_len
    """

    def get_enc_outdim(self):
        return self.main_layers[self.struct_enc[-1]].get_outdim()


def SeqToOneClassifier(nin, naux, nout, cfg, param=None, cfg_add=None, param_add=None, frz=False):
    if param:
        """
        if param_add:
            #cfg_add = MyConfig(cfg_add).get()
            model = SequentialClassifierDualEnc(
                    nin, nout, cfg, param, cfg_add, param_add, frz
                    )
            model_type = 'SequentialClassifierDualEnc'
        else:
            model = SequentialClassifierEnc(
                    nin, nout, cfg, param, frz
                    )
            model_type = 'SequentialClassifierEnc'
        """
        raise NotImplementedError
    else:
        model = SeqToOne(nin, nout, cfg, naux)
    return model
