import argparse
import os
import glob
import pandas as pd


def _parse():
    parser = argparse.ArgumentParser(description='generate text segments')
    parser.add_argument('segvideo_dir', type=str,
                        help='CMU-MOSEI segmented video dir')
    parser.add_argument('meta', type=str, help='output meta file')
    args = parser.parse_args()
    return args


def _main():
    args = _parse()
    os.makedirs(os.path.dirname(args.meta), exist_ok=True)
    
    lists = [[d.strip(os.sep).split(os.sep)[-1], 0] for i, d in enumerate(glob.glob(os.path.join(args.segvideo_dir, '*')))]
    df = pd.DataFrame(lists)
    df.to_csv(args.meta, header=['Class_ID', 'class'], index=None)


if __name__ == '__main__':
    _main()
