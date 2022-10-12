import os
import glob
import argparse
import numpy as np
import pdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=2)
sns.set_style('whitegrid')
sns.set_palette('Set3')
plt.rcParams["font.size"] = 28


def _parse():
    parser = argparse.ArgumentParser(description='open hdf5 data')
    parser.add_argument('ind', type=str, help='input attention dir')
    parser.add_argument('outf', type=str, help='output violin plot')
    parser.add_argument('-l', type=str, nargs='*', help='graph label')
    return parser.parse_args()


"""
npz key: 'att_video', 'att_audio', 'att_text', 'att_dec'
"""
def get_modal_weights(ind):
    modal_weights = []
    for npz in glob.glob(os.path.join(ind, '*.npz')):
        s = np.load(npz)
        modal_weights.append(s['att_dec'])
    return np.stack(modal_weights)


def plot_modal_weight(ind, outf, label=[]):
    os.makedirs(os.path.dirname(outf), exist_ok=True)

    modal_weights = get_modal_weights(ind)

    df = pd.DataFrame({
        'video':modal_weights[:, 0],
        'audio':modal_weights[:, 1],
        'lang.':modal_weights[:, 2],
        })
    df_melt = pd.melt(df)
    df_melt['input'] = label[0] if label else 'input'

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    """
    sns.violinplot(
            x='variable', y='value', data=df_melt, hue='input',
            dodge=True, jitter=True, color='black', palette='Set3', ax=ax, cut=0
            )
    """
    sns.violinplot(
            x='variable', y='value', data=df_melt,
            dodge=True, jitter=True, color='black', palette=['#f4b183', '#9dc3e6', '#ffd966'], ax=ax, cut=0
            )
    ax.set_xlabel('')
    ax.set_ylabel('modality weight')
    ax.set_ylim(0, 1)
    #ax.set_title(ind.split(os.sep)[-2])
    #ax.legend()
    ax.legend([],[], frameon=False)
    plt.savefig(outf)


def _main():
    args = _parse()
    plot_modal_weight(args.ind, args.outf, args.l)


if __name__ == '__main__':
    _main()
