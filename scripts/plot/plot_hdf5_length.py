import os
import argparse
import numpy as np
import pdb
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt


def _parse():
    parser = argparse.ArgumentParser(description='open hdf5 data')
    parser.add_argument('inf', type=str, help='input hdf5 rootdir')
    parser.add_argument('outf', type=str, help='input faceimages rootdir')
    parser.add_argument('-v', action='store_true', help='show feature length')
    return parser.parse_args()


def _main():
    args = _parse()

    data_list = []
    with h5py.File(args.inf, 'r') as h:
        for k in tqdm(h['data']):
            x = h['data'][k][...]
            n = x.shape[0]
            data_list.append(n)
            if args.v:
                print(k, n)
   
    print('total loaded: {}\n'.format(len(data_list)))
    
    data = np.array(data_list)
    plt.hist(data)
    plt.savefig(args.outf)


if __name__ == '__main__':
    _main()
