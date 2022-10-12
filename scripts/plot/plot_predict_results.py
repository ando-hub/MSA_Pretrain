import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pd


plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.axisbelow'] = True


def _parse():
    parser = argparse.ArgumentParser(description='plot distrib')
    parser.add_argument('inf', type=str, help='input result.csv')
    parser.add_argument('--outd', type=str, help='output distrib dir')
    parser.add_argument('--ext', type=str, default='.png', help='output distrib format')
    return parser.parse_args()


def plot(x, y, outf):
    plt.figure(figsize=(2, 2), tight_layout=True)
    plt.tick_params(length=0)
    plt.scatter(x, y, zorder=2)
    plt.hlines(0, -3, 3, colors='gray', zorder=1)
    plt.vlines(0, -3, 3, colors='gray', zorder=1)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xticks([-3, -2, -1, 0, 1, 2, 3])
    plt.yticks([-3, -2, -1, 0, 1, 2, 3])
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.grid()
    plt.savefig(outf+'.png')
    plt.savefig(outf+'.eps')
    plt.close()


def calc_var(rd, outd=None, out_ext='.png'):
    ref_intra = []
    ref_total = []
    out_intra = []
    out_total = []
    for video_id, _rd in rd.items():
        x = np.array(_rd['ref'])
        y = np.array(_rd['out'])
        ref_intra.append(x-x.mean())
        ref_total.append(x)
        out_intra.append(y-y.mean())
        out_total.append(y)
        if outd:
            outf = os.path.join(outd, video_id)
            plot(x, y, outf)

    ref_intra_var = np.concatenate(ref_intra, axis=0).var()
    ref_total_var = np.concatenate(ref_total, axis=0).var()
    out_intra_var = np.concatenate(out_intra, axis=0).var()
    out_total_var = np.concatenate(out_total, axis=0).var()
    print('Ref: [Total, Intra] = {:.4f}, {:.4f}'.format(ref_total_var, ref_intra_var))
    print('Out: [Total, Intra] = {:.4f}, {:.4f}'.format(out_total_var, out_intra_var))


def _main():
    args = _parse()
    if args.outd:
        os.makedirs(args.outd, exist_ok=True)

    df = pd.read_csv(args.inf)
    rd = {}
    for _, row in df[['uid', 'ref_regress', 'out_regress']].iterrows():
        video_id = '_'.join(row['uid'].split('_')[:-1])
        if video_id not in rd:
            rd[video_id] = {'ref':[], 'out':[]}
        rd[video_id]['ref'].append(row['ref_regress'])
        rd[video_id]['out'].append(row['out_regress'])

    calc_var(rd, args.outd)


if __name__ == '__main__':
    _main()
