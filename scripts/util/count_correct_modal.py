import os
import argparse
import glob
import pdb
import pandas as pd


def _parse():
    parser = argparse.ArgumentParser(description='count correct modal')
    parser.add_argument('inf', type=str, help='input result.csv')
    return parser.parse_args()


def _main():
    args = _parse()
    df = pd.read_csv(args.inf)
    columns = [c for c in df.columns if c.startswith('corrected_')]
    for c in columns:
        print(df[c].value_counts(dropna=False))
    pdb.set_trace()


if __name__ == '__main__':
    _main()

