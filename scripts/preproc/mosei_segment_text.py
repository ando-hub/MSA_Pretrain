import argparse
import os
import glob
from collections import defaultdict
from tqdm import tqdm


def _parse():
    parser = argparse.ArgumentParser(description='get utterance-level text from transcriptions')
    parser.add_argument('cmumosei_txt_dir', type=str,
                        help='${mosei_root}/Transcript/Segmented/Combined')
    parser.add_argument('interval_list', type=str, help='CMU-MOSEI interval list')
    parser.add_argument('output_dir', type=str, help='output segmented text dir')
    args = parser.parse_args()
    return args


def segmentation(txtf, outd, valid_segids=[]):
    for l in open(txtf):
        fname, segid, t_st, t_en, body = l.strip().split('___', 4)
        if valid_segids and segid not in valid_segids:
            continue
        outf = os.path.join(outd, fname + '_{}.txt'.format(segid))
        with open(outf, 'w') as fp:
            fp.write(body)
    return


def load_valid_dict(interval_list):
    valid_dict = defaultdict(list)
    for l in open(interval_list):
        fname, segid, _, _ = l.strip().split()
        valid_dict[fname].append(segid)
    return valid_dict


def _main():
    args = _parse()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.interval_list:
        valid_dict = load_valid_dict(args.interval_list)
    else:
        valid_dict = {}

    txtfs = glob.glob(os.path.join(args.cmumosei_txt_dir, '*.txt'))
    assert len(txtfs), 'No exist *.txt in {}'.format(args.cmumosei_text_dir)

    for txtf in tqdm(txtfs):
        fname = os.path.splitext(os.path.basename(txtf))[0]
        if valid_dict:
            if fname in valid_dict:
                segmentation(txtf, args.output_dir, valid_segids=valid_dict[fname])
        else:
            segmentation(txtf, args.output_dir)


if __name__ == '__main__':
    _main()
