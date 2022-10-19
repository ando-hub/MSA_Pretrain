import argparse
import os
from os.path import join, splitext, exists
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.simplefilter('ignore', UndefinedMetricWarning)


def get_classification_results(df, task):
    outstr = []

    _df = df.dropna(subset=['actu_{}'.format(task)])
    y_true = _df['actu_{}'.format(task)].values
    y_pred = _df['pred_{}'.format(task)].values
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    cm = confusion_matrix(y_true, y_pred)
    rslt = classification_report(y_true, y_pred)
    rslt_dic = classification_report(y_true, y_pred, output_dict=True)
    wacc = rslt_dic['accuracy']
    uacc = rslt_dic['macro avg']['recall']
    uacc = rslt_dic['macro avg']['recall']
    mf1 = rslt_dic['macro avg']['f1-score']
    wf1 = f1_score(y_true, y_pred, average='weighted')

    outstr.append('[task: {}] WA={:.4f} UA={:.4f} MF1={:.4f} WF1={:.4f}\n'.format(task, wacc, uacc, mf1, wf1))
    outstr.append(str(cm)+'\n')
    outstr.append(rslt+'\n')
    return outstr, (wacc, uacc, mf1, wf1)


def get_regression_results(df, task):
    outstr = []

    _df = df.dropna(subset=['ref_{}'.format(task)])
    y_true = _df['ref_{}'.format(task)].values
    y_pred = _df['out_{}'.format(task)].values
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = np.mean(np.absolute(y_true-y_pred))
    corr = np.corrcoef(y_pred, y_true)[0][1]

    outstr.append('[task: reg_{}] MAE={:.4f} Corr={:.4f}\n'.format(task, mae, corr))
    return outstr


def eval_performance(csvf, outf=None):
    df = pd.read_csv(csvf)
    classification_tasks = ['_'.join(v.split('_')[1:]) for v in df.columns if v.startswith('pred_')]
    regression_tasks = ['_'.join(v.split('_')[1:]) for v in df.columns if v.startswith('ref_')]

    outstr = []
    avg_value = []
    for _task in classification_tasks:
        _outstr, _scores = get_classification_results(df, _task)
        outstr.extend(_outstr)
        avg_value.append(_scores)
    if classification_tasks:
        avg_scores = np.array(avg_value).mean(axis=0).tolist()
        outstr.append('[task: overall] WA={:.4f} UA={:.4f} MF1={:.4f} WF1={:.4f}\n'.format(*avg_scores))

    for _task in regression_tasks:
        _outstr = get_regression_results(df, _task)
        outstr.extend(_outstr)

    outstr = '\n'.join(outstr)
    if outf:
        with open(outf, 'w') as fp:
            fp.write(outstr)
    else:
        print(csvf)
        print(outstr)


def _parse():
    parser = argparse.ArgumentParser(description=u'evaluate estimation performance')
    parser.add_argument('rootd', type=str, help='input result dir')
    parser.add_argument('-o', action='store_true', default=False,
                        help='overwrite result file (default: False)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse()

    for dpath, dname, fnames in os.walk(args.rootd):
        for fname in fnames:
            base, ext = splitext(fname)
            if ext != '.csv':
                continue

            outf = join(dpath, base+'.txt')
            if not args.o and exists(outf):
                continue
            eval_performance(join(dpath, fname), outf)
            print(outf)
