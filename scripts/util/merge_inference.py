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
    parser.add_argument('--merge-method', choices=['average', 'oracle'], default='average',
                        help='output result summary (csv)')
    return parser.parse_args()


def merge_inferences(fs, outf, merge_method):
    assert len(fs) == 3, 'merge files must be 3, but: '+','.join(fs)
    
    # df order: audio -> text -> video / each df is sorted by uids
    dfs = [pd.read_csv(f).sort_values('uid') for f in sorted(fs)]
    pred_classes = [c.split('_')[-1] for c in dfs[0].columns.values if c.startswith('pred_')]
    
    out_df = dfs[0].copy()
    for c in pred_classes:
        probs = [df['prob_'+c].to_numpy() for df in dfs]
        probs = np.stack(probs).T
        if merge_method == 'average':
            out_probs = np.mean(probs, axis=1)
            pdb.set_trace()
        elif merge_method == 'oracle':
            max_out_val = np.amax(probs, axis=1)
            min_out_val = np.amin(probs, axis=1)
            max_out_idx = np.argmax(probs, axis=1)
            min_out_idx = np.argmin(probs, axis=1)
            actu = out_df['actu_'+c].to_numpy()
            out_probs = min_out_val
            out_probs[actu == 1] = max_out_val[actu == 1]
            # save selected modal
            slct_idx = min_out_idx
            slct_idx[actu == 1] = max_out_idx[actu == 1]
            out_df['slct_'+c] = slct_idx
            # save correctly predicted modal
            preds = (probs >= 0.5)
            corrected_idx = []
            for i, a in enumerate(actu):
                corrected = np.where(preds[i] == a)[0]
                corrected = ','.join([str(v) for v in corrected])
                corrected_idx.append(corrected)
            out_df['corrected_'+c] = pd.Series(corrected_idx)
        out_preds = (out_probs >= 0.5).astype(np.int16)
        out_df['pred_'+c] = pd.Series(out_preds)
        out_df['prob_'+c] = pd.Series(out_probs)
    out_df.to_csv(outf)


if __name__ == '__main__':
    args = _parse()
    
    print('seek dirs to find inference results ...')
    result_dic = {}
    for dpath, dname, fnames in os.walk(args.rootd):
        for fname in fnames:
            if fname == 'result.csv':
                feat, model, setup = os.path.basename(dpath).split('.', 2)
                if feat in ['input_audio', 'input_video', 'input_text']:
                    key = '.'.join([model, setup])
                    if key not in result_dic:
                        result_dic[key] = []
                    result_dic[key].append(os.path.join(dpath, fname))
    
    print('start merging ...')
    for key, fs in result_dic.items():
        outd = os.path.join(args.outd, 'input_{}.'.format(args.merge_method)+key)
        os.makedirs(outd, exist_ok=True)
        outf = os.path.join(outd, 'result.csv')
        merge_inferences(fs, outf, args.merge_method)

        result_summary = os.path.join(outd, 'result.txt')
        eval_performance(outf, result_summary)
    
