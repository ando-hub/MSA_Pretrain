import os
import argparse
import numpy as np
import pdb
import h5py


def _parse():
    parser = argparse.ArgumentParser(description='open hdf5 data')
    parser.add_argument('inf', type=str, help='input faceimages rootdir')
    return parser.parse_args()


def _main():
    args = _parse()

    data_list = []
    with h5py.File(args.inf, 'r') as h:
        for k in h:
            print(k)
            for d in h[k]:
                print('\t', d, h[k][d])
                data_list.append(d)
                """
                for d2 in h[k][d]:
                    print(d, d2.shape)
                """
    
    print('total loaded: {}\n'.format(len(data_list)))
    pdb.set_trace()
    i = 1
    print('{}-th data: {}'.format(i, data_list[i]))
    x = _load(args.inf, data_list[i])
    print(x)
    print(x.shape)

def _load(inf, fid):
    with h5py.File(inf, 'r') as h:
        return h['data'][fid][...]

if __name__ == '__main__':
    _main()
