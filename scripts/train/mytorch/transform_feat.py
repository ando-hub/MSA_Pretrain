# coding:utf-8

import numpy as np
#from . import spec_augment
from .spec_augment import spec_augment


class SpecTransformer(object):
    """Transform Spectrogram

    :param str normf: normalization file
    :param int spaug_max_time_warp: SpecAugment time warping parameter
    :param int spaug_max_freq_width: SpecAugment frequency masking parameter
    :param int spaug_n_freq_mask: SpecAugment frequency masking parameter
    :param int spaug_max_time_width: SpecAugment time masking parameter
    :param int spaug_n_time_mask: SpecAugment time masking parameter
    :param int spaug_max_time_rate: SpecAugment time masking parameter
    """
    def __init__(
            self, normf=None,
            max_time_warp=0,
            max_freq_width=0, n_freq_mask=0,
            max_time_width=0, n_time_mask=0, max_time_rate=0,
            iseval=False,
            ):
        self.spec_augment_conf = {
                'resize_mode': 'PIL',
                'max_time_warp': max_time_warp,
                'max_freq_width': max_freq_width,
                'n_freq_mask': n_freq_mask,
                'max_time_width': max_time_width,
                'n_time_mask': n_time_mask,
                'max_time_rate': max_time_rate,
                'inplace': False,
                'replace_with_zero': True
                }
        self.normalizer = ZNormalize(normf)
        self.iseval = iseval

    def __call__(self, x):
        x = self.normalizer(x)
        if not self.iseval:
            x = spec_augment(x, **self.spec_augment_conf)
            # x = spec_augment.spec_augment(x, **self.spec_augment_conf)
        return x


class ZNormalize(object):
    def __init__(self, normf=None):
        self.normf = normf
        if normf:
            self.m, self.s = self.__load_norm(normf)

    def __load_norm(self, normf):
        m = []
        s = []
        for l in open(normf):
            mi, si = l.strip().split('\t')
            m.append(float(mi))
            s.append(float(si))
        m = np.asarray(m, dtype=np.float32)
        s = np.asarray(s, dtype=np.float32)
        s[s == 0] = 1.
        return (m, s)

    def __call__(self, x):
        if self.normf:
            nlen = x.shape[0]
            x = (x-np.tile(self.m, (nlen, 1))) / np.tile(self.s, (nlen, 1))
        return x


def concat_feat(x, n_concat):
    if n_concat > 0:
        # expand input feature (copy 1st / final sample)
        _x = np.concatenate(
                (np.tile(x[0], (n_concat, 1)), x, np.tile(x[-1], (n_concat, 1))),
                axis=0)
        # concat consequtive samples
        y = np.zeros((x.shape[0], x.shape[1]*(n_concat*2+1)), dtype=x.dtype)
        for org_idx, i in enumerate(range(n_concat, x.shape[0]+n_concat)):
            y[org_idx] = _x[i-n_concat:i+n_concat+1].reshape(-1,)
        return y
    else:
        return x


def skip_feat(x, n_skip):
    return x[::n_skip+1]
