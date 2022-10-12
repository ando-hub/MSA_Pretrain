# coding:utf-8

import argparse
from test import child_func, get_logger

# import logging
# __logger = logging.getLogger('train')


def func1(a):
    __logger.debug('func1: {}'.format(a))
    __logger.info('func1: {}'.format(a))
    __logger.warning('func1: {}'.format(a))
    return


def _parse():
    parser = argparse.ArgumentParser(description=u'merge label.xlsx to json')
    parser.add_argument('-o', type=str, default=None, help='input logger file')
    parser.add_argument('-l', type=str, default='info', help='log level')
    return parser.parse_args()


def _main():
    args = _parse()
       
    __logger = get_logger(name='test', logf=args.o, level=args.l)
    
    func1('hoge')
    child_func('piyo')


if __name__ == '__main__':
    _main()

