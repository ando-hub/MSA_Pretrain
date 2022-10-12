import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pdb


def _parse():
    parser = argparse.ArgumentParser(description='open hdf5 data')
    parser.add_argument('ind', type=str, help='input attention dir')
    parser.add_argument('outd', type=str, help='output violin plot')
    return parser.parse_args()


"""
npz key: 'att_video', 'att_audio', 'att_text', 'att_dec'
"""
def plot_modal_attention(inf, outf):
    os.makedirs(os.path.dirname(outf), exist_ok=True)
    s = np.load(inf)
    keys = list(s.keys())

    fig = plt.figure()
    if 'att_video' in keys:
        y = s['att_video']
        x = np.arange(0, y.shape[0])*(1/3.)
        valid_idx = (y.sum(axis=1) > 0)
        ax = fig.add_subplot(3, 1, 1)
        ax.plot(x[valid_idx], y[valid_idx])
        ax.set_xlabel('time')
        ax.set_ylabel('att_video')

    if 'att_audio' in keys:
        y = s['att_audio']
        x = np.arange(0, y.shape[0])*0.02
        valid_idx = (y.sum(axis=1) > 0)
        ax = fig.add_subplot(3, 1, 2)
        ax.plot(x[valid_idx], y[valid_idx])
        ax.set_xlabel('time')
        ax.set_ylabel('att_audio')

    if 'att_text' in keys:
        y = s['att_text']
        x = np.arange(0, y.shape[0])
        valid_idx = (y.sum(axis=1) > 0)
        ax = fig.add_subplot(3, 1, 3)
        ax.plot(x[valid_idx], y[valid_idx])
        ax.set_xlabel('index')
        ax.set_ylabel('att_text')
    plt.savefig(outf)
    plt.close()


def _main():
    args = _parse()
    for inf in glob.glob(os.path.join(args.ind, '*.npz')):
        outf = os.path.join(args.outd, os.path.splitext(os.path.basename(inf))[0]+'.png')
        if not os.path.exists(outf):
            plot_modal_attention(inf, outf)


if __name__ == '__main__':
    _main()
