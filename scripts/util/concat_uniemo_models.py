# coding:utf-8

import argparse
import os
import csv
import pdb
import pandas as pd
import numpy as np
from mytorch.eval_performance import eval_performance


def _parse():
    parser = argparse.ArgumentParser(description=u'evaluate estimation performance')
    parser.add_argument('rootd', type=str, help='input result dir')
    parser.add_argument('outd', type=str, help='output merged result dir')
    return parser.parse_args()


def concat_uniemo_inferences(fs, outf):
    assert len(fs) == 6, 'unifying files must be 6, but: '+','.join(fs)
    
    # df order: audio -> text -> video / each df is sorted by uids
    dfs = []
    for i, f in enumerate(sorted(fs)):
        df = pd.read_csv(f).sort_values('uid')
        df = df.rename(columns={'actu_0': 'actu_'+str(i), 'pred_0': 'pred_'+str(i), 'prob_0': 'prob_'+str(i)})
        dfs.append(df)
    
    out_df = pd.concat(dfs, axis=1, join='outer')
    out_df.to_csv(outf)


if __name__ == '__main__':
    args = _parse()
    
    print('seek dirs to find inference results ...')
    result_dic = {}
    for dpath, dname, fnames in os.walk(args.rootd):
        for fname in fnames:
            task = dpath.split(os.sep)[-3]
            if fname == 'result.csv' and task.startswith('emo'):
                key = os.path.basename(dpath)
                if key not in result_dic:
                    result_dic[key] = []
                result_dic[key].append(os.path.join(dpath, fname))
  
    print('start merging ...')
    for key, fs in result_dic.items():
        outd = os.path.join(args.outd, key)
        os.makedirs(outd, exist_ok=True)
        outf = os.path.join(outd, 'result.csv')
        concat_uniemo_inferences(fs, outf)

        result_summary = os.path.join(outd, 'result.txt')
        eval_performance(outf, result_summary)
    
