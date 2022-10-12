import re
import argparse
import os
import csv
import pdb


def _parse():
    parser = argparse.ArgumentParser(description=u'evaluate estimation performance')
    parser.add_argument('rootd', type=str, help='input result dir')
    parser.add_argument('outf', type=str, help='output result summary (csv)')
    parser.add_argument('--earlystop-only', action='store_true', help='save earlystop result')
    parser.add_argument('--task', type=str, default='overall', help='evaluation task')
    parser.add_argument('--metric', type=str, nargs='*', default=['MF1'], help='evaluation metric')
    return parser.parse_args()


def _load_result(resultf):
    rd = {}
    for l in open(resultf):
        if l.startswith('[task:'):
            _, t, vals = l.strip().split(' ', 2)
            t = t.strip(']')
            vals = [float(v.split('=')[-1]) for v in vals.split()]
            rd[t] = {'WA': vals[0], 'UA': vals[1], 'MF1': vals[2], 'WF1': vals[3]}
    return rd


def select_best_result(rds, task, metric):
    best_fname = None
    best_val = 0.
    best_rd = {}
    for fname, rd in rds.items():
        rd_dev = rd['dev']
        rd_tst = rd['tst']
        if not best_fname:
            best_fname = fname
            best_rd = rd_tst
        else:
            _vals = [v for m, v in rd_dev[task].items() if m in metric]
            val = sum(_vals)/len(_vals)
            if best_val < val:
                best_fname = fname
                best_val = val
                best_rd = rd_tst
    return best_fname, best_rd


metric_order = ['WA', 'UA', 'MF1', 'WF1']
def rd2list(rd):
    return ['{:.4f}'.format(vd[m]) for t, vd in sorted(rd.items()) for m in metric_order]


if __name__ == '__main__':
    args = _parse()
    if args.earlystop_only:
        pattern = r'result\.tst\.txt'
    else:
        pattern = r'result\.tst\..*\.txt'

    result_dic = {}
    for dpath, dname, fnames in os.walk(args.rootd):
        if os.path.basename(dpath) != 'result':
            continue
        for fname in fnames:
            if re.match(pattern, fname):
                setup = (os.sep).join(dpath.replace(args.rootd, '').split(os.sep)[:-1])
                if setup not in result_dic:
                    result_dic[setup] = {}
                tstf = os.path.join(dpath, fname)
                devf = tstf.replace('tst', 'dev')
                result_dic[setup][fname] = {'dev': _load_result(devf), 'tst': _load_result(tstf)}

    # show results
    out_lines = []
    for setup, rds in sorted(result_dic.items()):
        fname, rd = select_best_result(rds, args.task, args.metric)
        out_list = setup.split(os.sep) + [fname] + rd2list(rd)
        out_lines.append(out_list)

    with open(args.outf, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(out_lines)
