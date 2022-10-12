"""
Spec Augment module for preprocessing i.e., data augmentation
See https://arxiv.org/pdf/1904.08779.pdf
"""

import numpy as np
import random

from PIL import Image
from PIL.Image import Resampling

np.random.seed(0)
random.seed(0)

def time_warp(x, max_time_warp=5, inplace=False, mode="PIL"):
    """time warp for spec augment

    move random center frame by the random width ~ uniform(-window, window)
    :param numpy.ndarray x: spectrogram (time, freq)
    :param int max_time_warp: maximum time frames to warp
    :param bool inplace: overwrite x with the result
    :param str mode: "PIL" (default, fast, not differentiable) or "sparse_image_warp" (slow, differentiable)
    :returns numpy.ndarray: time warped spectrogram (time, freq)
    """
    window = max_time_warp
    if mode == "PIL":
        t = x.shape[0]
        if t - window <= window:
            return x
        # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
        center = random.randrange(window, t - window)
        warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1

        left = Image.fromarray(x[:center]).resize((x.shape[1], warped), Resampling.BICUBIC)
        right = Image.fromarray(x[center:]).resize((x.shape[1], t - warped), Resampling.BICUBIC)
        if inplace:
            x[:warped] = left
            x[warped:] = right
            return x
        return np.concatenate((left, right), 0)
    else:
        raise NotImplementedError("mode supports only PIL")


def freq_mask(x, F=13, n_mask=1, replace_with_zero=False, inplace=False):
    """freq mask for spec agument

    :param numpy.ndarray x: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = x
    else:
        cloned = x.copy()
    num_channels = cloned.shape[1]

    for f in np.random.randint(0, F, n_mask):
        f_zero = random.randrange(0, num_channels-f)
        if replace_with_zero:
            cloned[:, f_zero:f_zero+f] = 0
        else:
            cloned[:, f_zero:f_zero+f] = cloned.mean()
    return cloned


def time_mask(spec, T=100, n_mask=1, max_mask_rate=0.2, replace_with_zero=False, inplace=False):
    """freq mask for spec agument

    :param numpy.ndarray spec: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = spec
    else:
        cloned = spec.copy()
    max_length = cloned.shape[0]

    for t in np.random.randint(0, T, n_mask):
        t = min([t, int(np.floor(max_length*max_mask_rate))])
        t_zero = random.randrange(0, max_length-t)
        if replace_with_zero:
            cloned[t_zero:t_zero+t] = 0
        else:
            cloned[t_zero:t_zero+t] = cloned.mean()
    return cloned


def spec_augment(x, resize_mode="PIL", max_time_warp=0,
                 max_freq_rate=0.1, n_freq_mask=2,
                 max_time_rate=0.1, n_time_mask=2, total_max_time_rate=0.2,
                 inplace=False, replace_with_zero=False):
    x_dim = len(x.shape)
    if x_dim == 2:
        return _spec_augment(x, resize_mode, max_time_warp,
                max_freq_rate, n_freq_mask,
                max_time_rate, n_time_mask, total_max_time_rate,
                inplace, replace_with_zero)
    elif x_dim == 3:
        return np.stack([
                _spec_augment(_x, resize_mode, max_time_warp,
                max_freq_rate, n_freq_mask,
                max_time_rate, n_time_mask, total_max_time_rate,
                inplace, replace_with_zero)
                for _x in x]
                )
    else:
        raise ValueError('input must be either 2 or 3-d array')


def _spec_augment(x, resize_mode="PIL", max_time_warp=0,
                 max_freq_rate=0.1, n_freq_mask=2,
                 max_time_rate=0.1, n_time_mask=2, total_max_time_rate=0.2,
                 inplace=False, replace_with_zero=False):
    """spec agument

    apply random time warping and time/freq masking
    default setting is based on LD (Librispeech double) in Table 2 https://arxiv.org/pdf/1904.08779.pdf

    :param numpy.ndarray x: (time, freq)
    :param str resize_mode: "PIL" (fast, nondifferentiable) or "sparse_image_warp" (slow, differentiable)
    :param int max_time_warp: maximum frames to warp the center frame in spectrogram (W)
    :param int freq_mask_width: maximum width of the random freq mask (F)
    :param int n_freq_mask: the number of the random freq mask (m_F)
    :param int time_mask_width: maximum width of the random time mask (T)
    :param int n_time_mask: the number of the random time mask (m_T)
    :param bool inplace: overwrite intermediate array
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    assert x.ndim == 2
    if max_time_warp > 0:
        x = time_warp(x, max_time_warp, inplace=inplace, mode=resize_mode)
    if max_freq_rate > 0 and n_freq_mask > 0:
        max_freq_width = int(np.floor(x.shape[1]*max_freq_rate))
        if max_freq_width > 0:
            x = freq_mask(
                    x, max_freq_width, n_freq_mask,
                    inplace=inplace,
                    replace_with_zero=replace_with_zero
                    )
    if max_time_rate > 0 and n_time_mask > 0:
        max_time_width = int(np.floor(x.shape[0]*max_time_rate))
        if max_time_width > 0:
            x = time_mask(
                    x, max_time_width, n_time_mask, total_max_time_rate,
                    inplace=inplace,
                    replace_with_zero=replace_with_zero
                    )
    return x


if __name__ == '__main__':
    x = np.random.rand(20, 30).astype(np.float32)
    y = spec_augment(x, max_time_warp=2, replace_with_zero=True)

    z = np.random.rand(2, 20, 30).astype(np.float32)
    w = spec_augment(z, max_time_warp=2, replace_with_zero=True)

    import pdb; pdb.set_trace()
