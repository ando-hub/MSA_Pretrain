import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pd


def _parse():
    parser = argparse.ArgumentParser(description='plot accuracy curve')
    parser.add_argument('ind', type=str, help='input result dir')
    parser.add_argument('outf', type=str, help='output accuracy curve')
    parser.add_argument('--task', type=str, default='overall', help='evaluation task')
    return parser.parse_args()


def _load_result(resultf):
    rd = {}
    for l in open(resultf):
        if l.startswith('[task:'):
            _, t, vals = l.strip().split(' ', 2)
            t = t.strip(']')
            vals = [float(v.split('=')[-1]) for v in vals.split()]
            rd[t] = {'WA': vals[0], 'UA': vals[1], 'MF1': vals[2], 'WF1': vals[3]}
    return rd


"""
npz key: 'att_video', 'att_audio', 'att_text', 'att_dec'
"""
def plot_acc_curve(ind, outf, task):
    os.makedirs(os.path.dirname(outf), exist_ok=True)

    # save results
    rdic = {}
    for f in glob.glob(os.path.join(ind, 'result.*.txt')):
        ep = int(os.path.basename(f).split('.')[1])
        _r = _load_result(f)
        rdic[ep] = _r[task]
    
    df = pd.DataFrame(rdic).T.sort_index()
    
    # plot accuracy curve
    cols = df.columns
    ncols = len(cols)
    
    fig = plt.figure()
    for i, v in enumerate(df.columns):
        ax = fig.add_subplot(ncols, 1, i+1)
        ax.plot(df[v].index, df[v].values)
        ax.set_xlabel('epoch')
        ax.set_ylabel(v)
        print('{} best: ep.{}'.format(v, df[v].idxmax()))
        print(rdic[df[v].idxmax()])
    plt.savefig(outf)
    plt.close()


def _main():
    args = _parse()
    plot_acc_curve(args.ind, args.outf, args.task)


if __name__ == '__main__':
    _main()
