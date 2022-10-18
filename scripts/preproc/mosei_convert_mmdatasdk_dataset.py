import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm


def _parse():
    parser = argparse.ArgumentParser(description='Create')
    parser.add_argument('inf', type=str, help='input hdf5')
    parser.add_argument('interval', type=str, help='input mosei_interval.txt')
    parser.add_argument('outd_video', type=str, help='output video feature dir')
    parser.add_argument('outd_audio', type=str, help='output audio feature dir')
    parser.add_argument('outd_text', type=str, help='output text feature dir')
    return parser.parse_args()


def get_fiddic(interval):
    fiddic = {}
    for l in open(interval):
        fid, n, st, en = l.strip().split()
        if fid not in fiddic:
            fiddic[fid] = {}
        fiddic[fid][(st, en)] = '_'.join((fid, n))
    return fiddic


def _main():
    args = _parse()
    os.makedirs(args.outd_video, exist_ok=True)
    os.makedirs(args.outd_audio, exist_ok=True)
    os.makedirs(args.outd_text, exist_ok=True)

    fiddic = get_fiddic(args.interval)

    with open(args.inf, 'rb') as fp:
        data = pickle.load(fp)

    for vset, _d in data.items():
        print('unpacking subset: {}'.format(vset))
        for _id, _v, _a, _t in tqdm(zip(_d['id'], _d['vision'], _d['audio'], _d['text']), total=len(_d['id'])):
            video_name, st, en = _id
            clip_name = fiddic[video_name][(st, en)]
            np.save(os.path.join(args.outd_video, clip_name+'.npy'), _v)
            np.save(os.path.join(args.outd_audio, clip_name+'.npy'), _a)
            np.save(os.path.join(args.outd_text, clip_name+'.npy'), _t)


if __name__ == '__main__':
    _main()

