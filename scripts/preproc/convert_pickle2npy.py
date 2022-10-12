import os
import argparse
import pdb
import pickle
import numpy as np


def _parse():
    parser = argparse.ArgumentParser(description='Convert pickle to npy feature files')
    parser.add_argument('inf', type=str, help='input [align|noalign].pkl')
    parser.add_argument('outd_video', type=str, help='output video feature dir')
    parser.add_argument('outd_audio', type=str, help='output audio feature dir')
    parser.add_argument('outd_text', type=str, help='output text feature dir')
    return parser.parse_args()


def _main():
    args = _parse()
    os.makedirs(args.outd_video, exist_ok=True)
    os.makedirs(args.outd_audio, exist_ok=True)
    os.makedirs(args.outd_text, exist_ok=True)

    print('load pickle data ...')
    with open(args.inf, 'rb') as fp:
        data = pickle.load(fp)

    for _d in data.values():
        for _id, _v, _a, _t in zip(_d['id'], _d['vision'], _d['audio'], _d['text']):
            if isinstance(_id, str):
                fname = _id.replace('$_$', '_') + '.npy'
            else:
                if isinstance(_id[0], str):
                    fname = _id[0] + '.npy'
                else:
                    fname = _id[0].decode('utf-8') + '.npy'
            np.save(os.path.join(args.outd_video, fname), _v)
            np.save(os.path.join(args.outd_audio, fname), _a)
            np.save(os.path.join(args.outd_text, fname), _t)


if __name__ == '__main__':
    _main()
