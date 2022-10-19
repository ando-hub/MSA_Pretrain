import re
import argparse
import os


def _parse():
    parser = argparse.ArgumentParser(description=u'evaluate estimation performance')
    parser.add_argument('rootd', type=str, help='input result dir')
    parser.add_argument('outd', type=str, help='output result summary (csv)')
    return parser.parse_args()


def _load_result(scoref):
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
    return rd


def add_result(src_dic, resultf):
    if not os.path.exists(resultf):
        return

    result = _load_result(resultf)
    for task, valdic in result.items():
        if task not in src_dic:
            src_dic[task] = {}
        for k, v in valdic.items():
            if k not in src_dic[task]:
                src_dic[task][k] = []
            src_dic[task][k].append(v)


def get_avg_score_str(valdic, keys=[]):
    if keys:
        return ['{}={:.4f}'.format(k, sum(valdic[k])/len(valdic[k])) for k in keys]
    else:
        return ['{}={:.4f}'.format(k, sum(v)/len(v)) for k, v in valdic.items()]


cls_key = ['WA', 'UA', 'MF1', 'WF1']
reg_key = ['MAE', 'Corr']

if __name__ == '__main__':
    args = _parse()

    result_dic = {}
    for dpath, dname, fnames in os.walk(args.rootd):
        if os.path.basename(dpath) == 'result':
            setup = re.sub(r'seed[0-9]', '', dpath.split(os.sep)[-2])
            if setup == 'infer':
                continue
            if setup not in result_dic:
                result_dic[setup] = {'dev': {}, 'tst': {}}
            add_result(result_dic[setup]['dev'], os.path.join(dpath, 'result.dev.txt'))
            add_result(result_dic[setup]['tst'], os.path.join(dpath, 'result.tst.txt'))

    for setup, _rd in result_dic.items():
        print(setup, '# merged: {}'.format(len(_rd['tst']['reg_regress']['MAE'])))
        for dset, task_results in _rd.items():
            outf = os.path.join(args.outd, setup, 'result', 'result.{}.txt'.format(dset))
            os.makedirs(os.path.dirname(outf), exist_ok=True)
            out_lines = []
            for task, vals in task_results.items():
                if 'reg' in task:
                    score_str = get_avg_score_str(vals, reg_key)
                else:
                    score_str = get_avg_score_str(vals, cls_key)
                out_lines.append(' '.join(['[task:', task+']'] + score_str))
            with open(outf, 'w') as fp:
                fp.write('\n\n'.join(out_lines)+'\n')
