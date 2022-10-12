import argparse
import os
import glob
import pandas as pd
import h5py
import numpy as np
from cmu_mosi_std_folds import cmu_mosi_std_folds


def _parse():
    parser = argparse.ArgumentParser(description='create label file')
    parser.add_argument('label', type=str, help='CMU_MOSI_Opinion_Labels.csd')
    parser.add_argument('outd', type=str, help='output label dir')
    parser.add_argument('--label-format', choices=['sentiment_regress'],
                        default='sentiment_regress',
                        help='label format')
    args = parser.parse_args()
    return args


def _main():
    args = _parse()
    os.makedirs(args.outd, exist_ok=True)

    train_fold, valid_fold, test_fold = cmu_mosi_std_folds()
    label_dic = {'train':[], 'valid':[], 'test':[]}
    with h5py.File(args.label, 'r') as h:
        for fid in h['Opinion Segment Labels']['data']:
            if fid in train_fold:
                dset = 'train'
            elif fid in valid_fold:
                dset = 'valid'
            elif fid in test_fold:
                dset = 'test'
            else:
                continue

            data = h['Opinion Segment Labels']['data'][fid]
            for n, f in enumerate(data['features']):
                segid = n+1
                clip_id = fid+'_'+str(segid)
                if args.label_format == 'emo_class':
                    label = [str(v) for v in (f[1:] > 0).astype(np.int64)]
                elif args.label_format == 'sentiment_regress':
                    label = [str(f[0])]
                elif args.label_format == 'sentiment_class':
                    raise NotImplementedError
                label_dic[dset].append(' '.join([clip_id] + label))

    for fold, lines in label_dic.items():
        outf = os.path.join(args.outd, fold + '.txt')
        with open(outf, 'w') as fp:
            fp.write('\n'.join(lines)+'\n')


if __name__ == '__main__':
    _main()
