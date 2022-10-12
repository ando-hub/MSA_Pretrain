import os
import argparse
import numpy as np
import pdb
import h5py


def _parse():
    parser = argparse.ArgumentParser(description='open hdf5 data')
    parser.add_argument('inf', type=str, help='input hdf5')
    parser.add_argument('outd', type=str, help='output npy dir')
    return parser.parse_args()


def _main():
    args = _parse()
    os.makedirs(args.outd, exist_ok=True)

    with h5py.File(args.inf, 'r') as h:
        for k in h:
            for fid in h[k]:
                x = h['data'][fid][...]
                outf = os.path.join(args.outd, fid+'.npy')
                if not os.path.exists(outf):
                    np.save(outf, x)


if __name__ == '__main__':
    _main()
