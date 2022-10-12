import torch
import torch.nn as nn

from .layers import MyCNN, MyLSTM, StructuredSelfAttention, MyLinear, \
                    MyFlatten, MyLinearCond, MyLinearSlct
from .config import MyConfig
from pdb import set_trace


class SequentialClassifier(nn.Module):
    def __init__(self, nin, nout, cfg, naux=None):
        super(SequentialClassifier, self).__init__()
        self.struct = cfg['model']['struct']

        # save encoder path
        self.struct_enc = []
        for l in self.struct:
            self.struct_enc.append(l)
            if l.startswith('att'):
                break

        self.task = nout.keys()
        self.aux_embed = False

        # auxiliary input: linear
        if naux and naux > 0 and 'aux' in cfg:
            self.aux_layer = MyLinear(naux, **cfg['aux']['full'])
            nemb = self.aux_layer.get_outdim()
            self.output_type = cfg['aux']['output']['layer']
        else:
            self.output_type = 'full'

        layerdict = {}
        nin_ch = 1
        # set parameters
        for name in self.struct:
            _cfg = cfg['modelparam'][name] if name in cfg['modelparam'] else None
            if name.startswith('cnn'):
                _l = MyCNN(nin, nin_ch, **_cfg)
                nin_ch = _l.get_outch()
            elif name.startswith('lstm'):
                _l = MyLSTM(nin, **_cfg)
            elif name.startswith('att'):
                _l = StructuredSelfAttention(nin, **_cfg)
            elif name.startswith('linearcond') or name.startswith('fullcond'):
                _l = MyLinearCond(nin, nemb, **_cfg)
                self.aux_embed = True
            elif name.startswith('linearslct') or name.startswith('fullslct'):
                _l = MyLinearSlct(nin, naux, **_cfg)
            elif name.startswith('linear') or name.startswith('full'):
                _l = MyLinear(nin, **_cfg)
            elif name.startswith('flat'):
                _l = MyFlatten(nin, nin_ch)
            else:
                raise NotImplementedError('invalid layer name: {}'.format(name))

            layerdict[name] = _l
            nin = _l.get_outdim()

        # final output: linear (multi-class)
        for _task, _nout in nout.items():
            name = 'out_{}'.format(_task)
            if self.output_type in ['linear', 'full']:
                _l = MyLinear(nin, _nout, nlay=1, activ=None, drop=0.)
            elif self.output_type in ['linearcond', 'fullcond']:
                _l = MyLinearCond(nin, nemb, _nout, activ=None, drop=0., batchnorm=False)
                self.aux_embed = True
            elif self.output_type in ['linearslct', 'fullslct']:
                _l = MyLinearSlct(nin, naux, _nout, activ=None, drop=0., batchnorm=False)
            layerdict[name] = _l

        self.layerdict = nn.ModuleDict(layerdict)

        # initialize
        for name, param in self.named_parameters():
            param.data.normal_(0., 0.01)
        return

    def forward(self, x, x_len, aux=None):
        if self.aux_embed:
            e, _ = self.aux_layer(aux)

        for name in self.struct:
            if name.startswith('linearcond') or name.startswith('fullcond'):
                x, x_len = self.layerdict[name](x, x_len, e)
            elif name.startswith('linearslct') or name.startswith('fullslct'):
                x, x_len = self.layerdict[name](x, x_len, aux)
            else:
                x, x_len = self.layerdict[name](x, x_len)

        y = {}
        for _task in self.task:
            if self.output_type in ['linear', 'full']:
                y[_task] = self.layerdict['out_{}'.format(_task)](x, x_len)[0]
            elif self.output_type in ['linearcond', 'fullcond']:
                y[_task] = self.layerdict['out_{}'.format(_task)](x, x_len, e)[0]
            elif self.output_type in ['linearslct', 'fullslct']:
                y[_task] = self.layerdict['out_{}'.format(_task)](x, x_len, aux)[0]
        return y

    def forward_enc(self, x, x_len):
        for name in self.struct_enc:
            x, x_len = self.layerdict[name](x, x_len)
        return x, x_len

    def get_enc_outdim(self):
        return self.layerdict[self.struct_enc[-1]].get_outdim()


class SequentialClassifierEnc(nn.Module):
    def __init__(self, nin, nout, cfg1, param1, frz=False):
        super(SequentialClassifierEnc, self).__init__()
        self.enc1 = SequentialClassifier(nin, nout, cfg1)
        # self.enc2 = SequentialClassifier(nin, nout, cfg2)

        # prepare decoder
        self.struct_dec = []
        dec_flg = False
        for l in cfg1['model']['struct']:
            if l.startswith('att'):
                dec_flg = True
            if dec_flg:
                self.struct_dec.append(l)
        self.struct_dec.pop(0)
        # overwrite decoder input dim. (enc1_outdim + enc2_outdim)
        # nin = self.enc1.get_enc_outdim() + self.enc2.get_enc_outdim()
        nin = self.enc1.get_enc_outdim()

        self.task = nout.keys()

        layerdict = {}
        nin_ch = 1
        # set parameters
        for name in self.struct_dec:
            _cfg = cfg1['modelparam'][name] if name in cfg1['modelparam'] else None
            if name.startswith('cnn'):
                _l = MyCNN(nin, nin_ch, **_cfg)
                nin_ch = _l.get_outch()
            elif name.startswith('lstm'):
                _l = MyLSTM(nin, **_cfg)
            elif name.startswith('att'):
                _l = StructuredSelfAttention(nin, **_cfg)
            elif name.startswith('linear') or name.startswith('full'):
                _l = MyLinear(nin, **_cfg)
            elif name.startswith('flat'):
                _l = MyFlatten(nin, nin_ch)

            layerdict[name] = _l
            nin = _l.get_outdim()

        # final output: linear (multi-class)
        for _task, _nout in nout.items():
            name = 'out_{}'.format(_task)
            _l = MyLinear(nin, _nout, nlay=1, activ=None, drop=0.)
            layerdict[name] = _l

        self.layerdict = nn.ModuleDict(layerdict)

        # initialize
        for name, param in self.named_parameters():
            param.data.normal_(0., 0.01)

        # load trained parameters
        self.enc1.load_state_dict(torch.load(param1), strict=False)
        #self.enc2.load_state_dict(torch.load(param2), strict=False)

        # parameter freezing
        if frz:
            for p in self.enc1.parameters():
                p.requires_grad = False
        return

    def forward(self, x, x_len):
        h, _ = self.enc1.forward_enc(x, x_len)
        for name in self.struct_dec:
            h, _ = self.layerdict[name](h, _)

        y = {}
        for _task in self.task:
            y[_task] = self.layerdict['out_{}'.format(_task)](h, _)[0]
        return y


class SequentialClassifierDualEnc(nn.Module):
    def __init__(self, nin, nout, cfg1, param1, cfg2, param2, frz=False):
        super(SequentialClassifierDualEnc, self).__init__()
        self.enc1 = SequentialClassifier(nin, nout, cfg1)
        self.enc2 = SequentialClassifier(nin, nout, cfg2)

        # prepare decoder
        self.struct_dec = []
        dec_flg = False
        for l in cfg1['model']['struct']:
            if l.startswith('att'):
                dec_flg = True
            if dec_flg:
                self.struct_dec.append(l)
        self.struct_dec.pop(0)
        # overwrite decoder input dim. (enc1_outdim + enc2_outdim)
        nin = self.enc1.get_enc_outdim() + self.enc2.get_enc_outdim()

        self.task = nout.keys()

        layerdict = {}
        nin_ch = 1
        # set parameters
        for name in self.struct_dec:
            _cfg = cfg1['modelparam'][name] if name in cfg1['modelparam'] else None
            if name.startswith('cnn'):
                _l = MyCNN(nin, nin_ch, **_cfg)
                nin_ch = _l.get_outch()
            elif name.startswith('lstm'):
                _l = MyLSTM(nin, **_cfg)
            elif name.startswith('att'):
                _l = StructuredSelfAttention(nin, **_cfg)
            elif name.startswith('linear') or name.startswith('full'):
                _l = MyLinear(nin, **_cfg)
            elif name.startswith('flat'):
                _l = MyFlatten(nin, nin_ch)

            layerdict[name] = _l
            nin = _l.get_outdim()

        # final output: linear (multi-class)
        for _task, _nout in nout.items():
            name = 'out_{}'.format(_task)
            _l = MyLinear(nin, _nout, nlay=1, activ=None, drop=0.)
            layerdict[name] = _l

        self.layerdict = nn.ModuleDict(layerdict)

        # initialize
        for name, param in self.named_parameters():
            param.data.normal_(0., 0.01)

        # load trained parameters
        self.enc1.load_state_dict(torch.load(param1), strict=False)
        self.enc2.load_state_dict(torch.load(param2), strict=False)

        # parameter freezing
        if frz:
            for p in self.enc1.parameters():
                p.requires_grad = False
            for p in self.enc2.parameters():
                p.requires_grad = False
        return

    def forward(self, x, x_len):
        h1, _ = self.enc1.forward_enc(x, x_len)
        h2, _ = self.enc2.forward_enc(x, x_len)
        h = torch.cat((h1, h2), 1)
        for name in self.struct_dec:
            h, _ = self.layerdict[name](h, _)

        y = {}
        for _task in self.task:
            y[_task] = self.layerdict['out_{}'.format(_task)](h, _)[0]
        return y


def load_model(nin, naux, nout, cfg, param=None, cfg_add=None, param_add=None, frz=False):
    if param:
        if param_add:
            cfg_add = MyConfig(cfg_add).get()
            model = SequentialClassifierDualEnc(
                    nin, nout, cfg, param, cfg_add, param_add, frz
                    )
            model_type = 'SequentialClassifierDualEnc'
        else:
            model = SequentialClassifierEnc(
                    nin, nout, cfg, param, frz
                    )
            model_type = 'SequentialClassifierEnc'
    else:
        model = SequentialClassifier(nin, nout, cfg, naux)
        model_type = 'SequentialClassifier'
    return model, model_type
