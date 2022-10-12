import os
import argparse
import glob
import pdb
import numpy as np

def _parse():
    parser = argparse.ArgumentParser(description='count label')
    parser.add_argument('ind', type=str, help='input label dir')
    return parser.parse_args()


def _main():
    args = _parse()

    outl = []
    for f in glob.glob(os.path.join(args.ind, '*txt')):
        _outl = []
        for l in open(f):
            _outl.append([int(v) for v in l.strip().split()[1:]])
        x = np.array(_outl)
        print(os.path.basename(f), x.shape)
        print(x.sum(axis=0))
        outl.append(x)
    arr = np.concatenate(outl, axis=0)
    print('overall')
    print(arr.shape)
    print(arr.sum(axis=0))

if __name__ == '__main__':
    _main()

