import os
import argparse
import numpy as np
import pdb
import h5py
import pickle


def _parse():
    parser = argparse.ArgumentParser(description='open cmu-mosei pickle data')
    parser.add_argument('inf', type=str, help='input pickle')
    return parser.parse_args()


def _main():
    args = _parse()
    print('load data ...')
    with open(args.inf, 'rb') as fp:
        data = pickle.load(fp)
    
    print('data structure:')
    for key, val in data.items():
        print(key)
        for key2, val2 in val.items():
            print('\t', key2, val2.shape)
    pdb.set_trace()


if __name__ == '__main__':
    _main()
