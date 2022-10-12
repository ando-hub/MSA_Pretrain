import numpy as np
import glob
import os
import argparse
from tqdm import tqdm


def _parse():
    parser = argparse.ArgumentParser(description=u'merge label.xlsx to json')
    parser.add_argument('ind', type=str, help='input *.xlsx dir')
    parser.add_argument('outd', type=str, help='output.json')
    parser.add_argument('-r', type=int, default=10, help='subsampling rate (default: 10)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse()
    os.makedirs(args.outd, exist_ok=True)

    for inf in tqdm(glob.glob(os.path.join(args.ind, '*.npy'))):
        outf = os.path.join(args.outd, os.path.basename(inf))
        if not os.path.exists(outf):
            try:
                x = np.load(inf)
                np.save(outf, x[:, ::args.r, :])
            except Exception as e:
                print(e)
