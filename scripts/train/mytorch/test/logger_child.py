# coding:utf-8

from logging import getLogger

__logger = getLogger(__name__)


def child_func(a):
    __logger.debug('a: {}'.format(a))
    __logger.info('a: {}'.format(a))
    __logger.warn('a: {}'.format(a))
    return
