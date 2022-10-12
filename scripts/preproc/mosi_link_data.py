import os
import glob
import argparse


def _parse():
    parser = argparse.ArgumentParser(description='copy feature files')
    parser.add_argument("ind", type=str, help='CMU-MOSI input feature dir')
    parser.add_argument("outd", type=str, help='CMU-MOSI output feature dir')
    args = parser.parse_args()
    return args


def _main():
    args = _parse()
    os.makedirs(args.outd, exist_ok=True)
    
    for f in glob.glob(os.path.join(args.ind, '*')):
        outf = os.path.join(args.outd, os.path.basename(f))
        if os.path.exists(outf):
            os.unlink(outf)
        os.symlink(os.path.abspath(f), os.path.abspath(outf))


if __name__ == "__main__":
    _main()
