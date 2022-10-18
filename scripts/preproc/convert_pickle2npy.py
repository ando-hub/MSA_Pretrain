import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm


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

    for vset, _d in data.items():
        print('unpacking subset: {}'.format(vset))
        for _id, _v, _a, _t in tqdm(zip(_d['id'], _d['vision'], _d['audio'], _d['text']), total=len(_d['id'])):
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
