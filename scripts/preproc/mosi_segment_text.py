import argparse
import os
import glob
from tqdm import tqdm


def _parse():
    parser = argparse.ArgumentParser(description='generate text segments')
    parser.add_argument('cmumosi_txt_dir', type=str,
                        help='${mosi_root}/Transcript/Segmented')
    parser.add_argument('output_dir', type=str, help='output segmented text dir')
    args = parser.parse_args()
    return args


def segmentation(txtf, outd):
    fname = os.path.splitext(os.path.basename(txtf))[0]
    for l in open(txtf):
        segid, body = l.strip().split('_DELIM_')
        outf = os.path.join(outd, fname + '_{}.txt'.format(segid))
        with open(outf, 'w') as fp:
            fp.write(body.strip())
    return


def _main():
    args = _parse()
    os.makedirs(args.output_dir, exist_ok=True)

    txtfs = glob.glob(os.path.join(args.cmumosi_txt_dir, '*.annotprocessed'))
    if not len(txtfs):
        raise ValueError('No exist *.annotprocessed in {}'.format(args.cmumosi_txt_dir))

    for txtf in tqdm(txtfs):
        segmentation(txtf, args.output_dir)


if __name__ == '__main__':
    _main()
