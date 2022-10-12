import numpy as np
import pdb
from numpy.lib.stride_tricks import sliding_window_view


def _sliding_window(array, window_size, window_shift=1, proc='flatten'):
    assert isinstance(array, np.ndarray), 'numpy array only'
    assert len(array.shape) == 2, '2-D array only'
    n_len, n_dim = array.shape
    if proc == 'flatten':
        if n_len < window_size:
            array = np.concatenate(
                    (array, np.zeros((window_size-n_len, n_dim), dtype=array.dtype)), axis=0
                    )
        x = sliding_window_view(array, (window_size, n_dim))
        n_frames = x.shape[0]
        x = x.reshape(n_frames, -1)
    elif proc in ['mean', 'meanstd']:
        _window_size = min((window_size, n_len))
        x = sliding_window_view(array, (_window_size, n_dim))
        n_frames = x.shape[0]
        x = x.reshape(n_frames, -1, n_dim)
        if proc == 'mean':
            x = x.mean(axis=1)
        else:
            x = np.concatenate((x.mean(axis=1), x.std(axis=1)), axis=1)
    else:
        raise ValueError('proc must be either flatten, mean, or meanstd')
    return x[::window_shift]


def sliding_window(array, window_size, window_shift=1, proc='flatten'):
    array_dim = len(array_shape)
    if array_dim == 2:
        return _sliding_window(array, window_size, window_shift, proc)
    elif array_dim == 3:
        return np.stack([_sliding_window(a, window_size, window_shift, proc) for a in array])
    else:
        raise ValueError('input must be either 2-d or 3-d array')


if __name__ == '__main__':
    x = np.arange(40, dtype=np.float32).reshape(10, 4)
    window_size = 1
    window_shift = 1

    print('x:', x.shape)
    print(x)
    y = sliding_window(x, window_size, window_shift, 'flatten')
    z = sliding_window(x, window_size, window_shift, 'mean')
    w = sliding_window(x, window_size, window_shift, 'meanstd')
    print('flatten:', y.shape)
    print(y)
    print('mean:', z.shape)
    print(z)
    print('meanstd:', w.shape)
    print(w)
    pdb.set_trace()
