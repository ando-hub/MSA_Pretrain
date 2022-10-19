import os


def sec2hms(sec):
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    _retstr = '{}s'.format(s)
    if m or h:
        _retstr = '{}m'.format(m) + _retstr
    if h:
        _retstr = '{}h'.format(h) + _retstr
    return _retstr


def update_link(src, dst):
    if os.path.exists(dst):
        os.unlink(dst)
    os.symlink(os.path.abspath(src), os.path.abspath(dst))
