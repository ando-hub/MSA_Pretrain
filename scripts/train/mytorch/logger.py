import os
import logging


def getlogger(logfile, level, stream=True):
    logger = logging.getLogger('logger')

    level = level.lower()
    if level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif level == 'info':
        logger.setLevel(logging.INFO)
    elif level == 'warning':
        logger.setLevel(logging.WARNING)
    elif level == 'error':
        logger.setLevel(logging.ERROR)
    else:
        raise ValueError('invalid log level: {}'.format(level))

    if logfile:
        logdir = os.path.dirname(logfile)
        if logdir:
            os.makedirs(logdir, exist_ok=True)
        handler = logging.FileHandler(filename=logfile)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        if stream:
            handler2 = logging.StreamHandler()
            handler2.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
            logger.addHandler(handler2)
    else:
        handler2 = logging.StreamHandler()
        handler2.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler2)

    return logger
