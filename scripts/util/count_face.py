import os
import argparse
import glob
import pdb


def _parse():
    parser = argparse.ArgumentParser(description='Detect Speech Interval')
    parser.add_argument('ind', type=str, help='input jpeg dir')
    return parser.parse_args()


def _main():
    args = _parse()
    

    for d in glob.glob(os.path.join(args.ind, '*')):
        nfaces = []
        for f in sorted(glob.glob(os.path.join(d, '*nfaces'))):
            with open(f) as fp:
                nfaces.append(int(fp.read().strip()))
        if 1 not in set(nfaces):
            print(d, set(nfaces))


if __name__ == '__main__':
    _main()

