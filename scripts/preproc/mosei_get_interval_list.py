import argparse
import os
import glob
import pdb
from tqdm import tqdm
import h5py


def _parse():
    parser = argparse.ArgumentParser(description='get interval info from texts and labels')
    parser.add_argument('cmumosei_txt_dir', type=str,
                        help='${mosei_root}/Transcript/Segmented/Combined')
    parser.add_argument('cmumosei_label_csd', type=str, help='CMU_MOSEI_Labels.csd')
    parser.add_argument('output_interval', type=str,
                        help='output interval file')
    args = parser.parse_args()
    return args


def get_interval_from_label(csd):
    label_interval_dic = {}
    with h5py.File(csd, 'r') as h:
        for fid in h['All Labels']['data']:
            label_interval_dic[fid] = []
            for interval in h['All Labels']['data'][fid]['intervals']:
                label_interval_dic[fid].append(','.join([str(v) for v in interval]))
    return label_interval_dic


def segmentation(txtf, outd, valid_segids=[]):
    for l in open(txtf):
        fname, segid, t_st, t_en, body = l.strip().split('___', 4)
        if valid_segids and segid not in valid_segids:
            continue
        outf = os.path.join(outd, fname + '_{}.txt'.format(segid))
        with open(outf, 'w') as fp:
            fp.write(body)
    return


def get_interval_dict(mosei_interval):
    valid_dict = {}
    if not mosei_interval:
        return valid_dict

    for l in open(mosei_interval):
        fid, segid, st, en = l.strip().split()
        if fid not in valid_dict:
            valid_dict[fid] = []
        valid_dict[fid].append(segid)
    return valid_dict


def _main():
    args = _parse()
    assert os.path.exists(args.cmumosei_label_csd), 'no exist {}'.format(args.cmumosei_label_csd)
    assert os.path.exists(args.cmumosei_txt_dir), 'no exist {}'.format(args.cmumosei_txt_dir)

    os.makedirs(os.path.dirname(args.output_interval), exist_ok=True)

    label_interval_dic = get_interval_from_label(args.cmumosei_label_csd)

    outl = []
    for f in sorted(glob.glob(os.path.join(args.cmumosei_txt_dir, '*.txt'))):
        fid = os.path.splitext(os.path.basename(f))[0]
        if fid in label_interval_dic:
            for l in open(f):
                fid, n, st, en, _ = l.strip().split('___', 4)
                if st == '0':
                    st = '0.0'
                if ','.join((st, en)) in label_interval_dic[fid]:
                    outl.append(' '.join((fid, n, st, en)))

    with open(args.output_interval, 'w') as fp:
        fp.write('\n'.join(sorted(outl)) + '\n')


if __name__ == '__main__':
    _main()
