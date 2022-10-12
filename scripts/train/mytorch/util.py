import os
import time


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


def count_load_time(dataset, N=100):
    start = time.time()
    if len(dataset) < N:
        N = len(dataset)
    for i in range(N):
         _ = train_set[i]
    proc_time = time.time()-start
    print('sample load time: {:.1f}s/sample ({:.1f}/{}sample)'.format(proc_time/N, proc_time, N))  


def update_link(src, dst):
    if os.path.exists(dst):
        os.unlink(dst)
    os.symlink(os.path.abspath(src), os.path.abspath(dst))
