import os
import glob
import argparse
import numpy as np
import pdb
import h5py
import torch
import pickle
import torchaudio
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

    for _d in data.values():
        for _id, _v, _a, _t in zip(_d['id'], _d['vision'], _d['audio'], _d['text']):
            video_name, st, en = _id
            clip_name = fiddic[video_name][(st, en)]
            np.save(os.path.join(args.outd_video, clip_name+'.npy'), _v)
            np.save(os.path.join(args.outd_audio, clip_name+'.npy'), _a)
            np.save(os.path.join(args.outd_text, clip_name+'.npy'), _t)

    """
    with h5py.File(args.inf, 'r') as h5in:
        with h5py.File(args.outf, 'w') as h5out:
            data_group = h5out.create_group('data')

            for k in h5in:
                for fid in tqdm(h5in[k]['data']):
                    # skip the files that are not included in the dataset
                    if fid not in fiddic:
                        continue
                    feat = h5in[k]['data'][fid]['features'][...]
                    interval = h5in[k]['data'][fid]['intervals'][...]
                    for seg_time, clip_id in fiddic[fid].items():
                        st, en = seg_time
                        idx = (st <= interval[:, 0]) & (interval[:, 1] <= en)
                        _feat = feat[idx]
                        data_group.create_dataset(name=clip_id, data=_feat, compression='gzip')
    """


if __name__ == '__main__':
    _main()

