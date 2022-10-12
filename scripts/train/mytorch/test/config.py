# coding:utf-8

import json
from os import path, makedirs


def load_config(f):
    with open(f) as fp:
        return json.load(fp)


def save_config(config, f):
    # prepare dir
    outd = path.dirname(f)
    if not path.exists(outd):
        makedirs(outd)
    # write
    with open(f, 'w') as fp:
        json.dump(config, fp, indent=2)
    return


class MyConfig(object):
    def __init__(self, configf=None):
        self.__dic = {}
        if configf:
            with open(configf) as fp:
                self.__dic = json.load(fp)

    def get(self):
        return self.__dic

    def set(self, d):
        self.__dic = d

    def set_args(self, args):
        _d = vars(args)
        for k, v in _d.items():
            if k.lower() in ['c', 'cfg', 'conf', 'config'] or v is None:
                continue
            self.__dic[k] = v
        return

    def write(self, outf):
        # prepare dir
        outd = path.dirname(outf)
        if not path.exists(outd):
            makedirs(outd)
        # write
        with open(outf, 'w') as fp:
            json.dump(self.__dic, fp, indent=2)
        return


def _init_dict():
    d = {}
    d['feat'] = {}
    d['feat']['nskip'] = 0
    d['feat']['ncon'] = 0

    d['model'] = {}
    d['model']['struct'] = ['cnn1', 'cnn2', 'cnn3', 'flatten', 'lstm', 'att', 'full1']

    d['modelparam'] = {}
    d['modelparam']['cnn1'] = {}
    d['modelparam']['cnn1']['cnn_ch'] = 16
    d['modelparam']['cnn1']['cnn_ker'] = (16, 12)
    d['modelparam']['cnn1']['cnn_str'] = 2
    d['modelparam']['cnn1']['batchnorm'] = True
    d['modelparam']['cnn1']['activ'] = 'relu'
    d['modelparam']['cnn1']['pool'] = 'max'
    d['modelparam']['cnn1']['pool_ker'] = (2, 2)
    d['modelparam']['cnn2'] = {}
    d['modelparam']['cnn2']['cnn_ch'] = 24
    d['modelparam']['cnn2']['cnn_ker'] = (6, 4)
    d['modelparam']['cnn2']['cnn_str'] = 1
    d['modelparam']['cnn2']['batchnorm'] = True
    d['modelparam']['cnn2']['activ'] = 'relu'
    d['modelparam']['cnn2']['pool'] = 'max'
    d['modelparam']['cnn2']['pool_ker'] = (2, 2)
    d['modelparam']['cnn3'] = {}
    d['modelparam']['cnn3']['cnn_ch'] = 16
    d['modelparam']['cnn3']['cnn_ker'] = (4, 3)
    d['modelparam']['cnn3']['cnn_str'] = 1
    d['modelparam']['cnn3']['batchnorm'] = True
    d['modelparam']['cnn3']['activ'] = 'relu'
    d['modelparam']['cnn3']['pool'] = 'max'
    d['modelparam']['cnn3']['pool_ker'] = (2, 2)
    d['modelparam']['lstm'] = {}
    d['modelparam']['lstm']['nhid'] = 256
    d['modelparam']['lstm']['nlay'] = 1
    d['modelparam']['lstm']['bidirec'] = True
    d['modelparam']['lstm']['drop'] = 0.5
    d['modelparam']['att'] = {}
    d['modelparam']['att']['nhead'] = 1
    d['modelparam']['att']['L2weight'] = 0
    d['modelparam']['full1'] = {}
    d['modelparam']['full1']['nhid'] = 64
    d['modelparam']['full1']['nlay'] = 2
    d['modelparam']['full1']['activ'] = 'relu'
    d['modelparam']['full1']['drop'] = 0.5

    d['train'] = {}
    d['train']['seed'] = 0
    d['train']['mbsize'] = 6
    d['train']['maxep'] = 100
    d['train']['earlystop'] = 'loss'
    d['train']['optimization'] = 'Adam'
    d['train']['lr'] = 0.0005
    d['train']['lrstep'] = True
    d['train']['gradclip'] = 10.0
    d['train']['classweight'] = True

    d['soft'] = {}
    d['soft']['bias'] = 0.75

    d['mlee'] = {}
    d['mlee']['thresh'] = 0.2

    return d


def _main():
    outf = '../json/base.json'

    c = MyConfig()
    c.set(_init_dict())
    c.write(outf)


if __name__ == '__main__':
    _main()
