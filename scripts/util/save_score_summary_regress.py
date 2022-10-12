import re
import argparse
import os
import csv
import pdb


def _parse():
    parser = argparse.ArgumentParser(description=u'evaluate estimation performance')
    parser.add_argument('rootd', type=str, help='input result dir')
    parser.add_argument('outf', type=str, help='output result summary (csv)')
    parser.add_argument('--dataset', choices=['dev', 'tst'], default='tst',
                        help='target dataset: [dev|tst] (default: tst)')
    return parser.parse_args()


def _load_result_regress(scoref):
    rd = {}
    for l in open(scoref):
        if l.startswith('[task:'):
            _, t, vals = l.strip().split(' ', 2)
            t = t.strip(']')
            vals = [float(v.split('=')[-1]) for v in vals.split()]
            if 'reg_' in l:
                rd[t] = {'MAE': vals[0], 'Corr': vals[1]}
            else:
                rd[t] = {'WA': vals[0], 'UA': vals[1], 'MF1': vals[2], 'WF1': vals[3]}
    return [
            rd['reg_regress']['MAE'], rd['reg_regress']['Corr'],
            rd['2']['WA'], rd['nz2']['WA'],
            rd['2']['WF1'], rd['nz2']['WF1']
            ]


if __name__ == '__main__':
    args = _parse()
    target_file_name = 'result.{}.txt'.format(args.dataset)

    result_dic = {}
    for dpath, dname, fnames in os.walk(args.rootd):
        if os.path.basename(dpath) == 'result':
            if not os.path.exists(os.path.join(dpath, target_file_name)):
                continue
            setup = (os.sep).join(dpath.replace(args.rootd, '').split(os.sep)[:-1])
            result_dic[setup] = _load_result_regress(os.path.join(dpath, target_file_name))

    # show results
    out_lines = []
    for setup, scores in sorted(result_dic.items()):
        out_list = setup.split(os.sep) + scores
        out_lines.append(out_list)

    with open(args.outf, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(out_lines)
