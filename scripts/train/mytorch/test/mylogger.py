# coding:utf-8

import logging
from logging import getLogger, StreamHandler, FileHandler, Formatter


def get_logger(name='test', logf=None, level='info', write_mode='w'):
    __logger = getLogger(name)

    # define loglevel
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('invalid log level: {}'.format(level))

    # define format
    log_format = Formatter('[%(asctime)s][%(levelname)s] %(message)s')

    if logf:
        handler = FileHandler(logf, mode=write_mode)
    else:
        handler = StreamHandler()
    handler.setFormatter(log_format)
    __logger.addHandler(handler)
    __logger.setLevel(numeric_level)

    return __logger
