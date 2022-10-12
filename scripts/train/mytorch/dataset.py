import os
import glob
import torch
import numpy as np
import h5py
from mytorch.spec_augment import spec_augment
from mytorch.sliding_window import sliding_window


def str2label(v):
    if '.' in v:
        return float(v)
    else:
        return int(v)


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, filelist, video_data, audio_data, text_data, input_modal,
                 video_feat_opt, audio_feat_opt, text_feat_opt, iseval=False):
        super(MultimodalDataset, self).__init__()
        self.video_feat_opt = video_feat_opt
        self.audio_feat_opt = audio_feat_opt
        self.text_feat_opt = text_feat_opt
        self.iseval = iseval

        self.video_data = video_data if 'video' in input_modal else None
        self.audio_data = audio_data if 'audio' in input_modal else None
        self.text_data = text_data if 'text' in input_modal else None

        # load filelists
        self.labels = {}
        for l in open(filelist):
            l = l.strip().split()
            self.labels[l[0]] = [str2label(v) for v in l[1:]]

        fids = self.labels.keys()
        if self.video_data:
            fids = [fid for fid in fids if self.validate_fid(fid, self.video_data)]
        if self.audio_data:
            fids = [fid for fid in fids if self.validate_fid(fid, self.audio_data)]
        if self.text_data:
            fids = [fid for fid in fids if self.validate_fid(fid, self.text_data)]
        self.fids = sorted(fids)


    def validate_fid(self, fid, data):
        if os.path.isfile(data):
            raise NotImplementedError
        else:
            return os.path.exists(os.path.join(data, fid+'.npy'))


    def load_data(self, data, fid):
        if os.path.isfile(data):
            with h5py.File(self.data, 'r') as h5:
                x = h5['data'][fid][...]
        else:
            x = np.load(os.path.join(data, fid+'.npy'))
        return x

    def slice_layer(self, data, slice_index=-1):
        if len(data.shape) == 2:
            return data
        elif len(data.shape) == 3:
            if slice_index < 0:
                return data
            else:
                if slice_index >= data.shape[0]:
                    raise ValueError('invalid layer-slicing: total_layer {}, slice_index {}'.format(data.shape[0], slice_index))
                return data[slice_index]
        else:
            raise ValueError('invalid input feature tensor: ', data.shape)

    def __len__(self):
        return len(self.fids)

    def __getitem__(self, idx):
        fid = self.fids[idx]
        label = self.labels[fid]

        data_v, data_a, data_t = None, None, None
        if self.video_data:
            data_v = self.load_data(self.video_data, fid)
            data_v = self.slice_layer(data_v, self.video_feat_opt['slice']['layer'])
            if self.video_feat_opt['sliding_window']['window_size'] > 1:
                data_v = sliding_window(
                        data_v,
                        self.video_feat_opt['sliding_window']['window_size'],
                        self.video_feat_opt['sliding_window']['window_shift'],
                        self.video_feat_opt['sliding_window']['proc']
                        )
            if not self.iseval:
                data_v = spec_augment(data_v, **self.video_feat_opt['spec_augment'])

        if self.audio_data:
            data_a = self.load_data(self.audio_data, fid)
            data_a = self.slice_layer(data_a, self.audio_feat_opt['slice']['layer'])
            if self.audio_feat_opt['sliding_window']['window_size'] > 1:
                data_a = sliding_window(
                        data_a,
                        self.audio_feat_opt['sliding_window']['window_size'],
                        self.audio_feat_opt['sliding_window']['window_shift'],
                        self.audio_feat_opt['sliding_window']['proc']
                        )
            if not self.iseval:
                data_a = spec_augment(data_a, **self.audio_feat_opt['spec_augment'])

        if self.text_data:
            data_t = self.load_data(self.text_data, fid)
            data_t = self.slice_layer(data_t, self.text_feat_opt['slice']['layer'])
            if self.text_feat_opt['sliding_window']['window_size'] > 1:
                data_t = sliding_window(
                        data_t,
                        self.text_feat_opt['sliding_window']['window_size'],
                        self.text_feat_opt['sliding_window']['window_shift'],
                        self.text_feat_opt['sliding_window']['proc']
                        )
            if not self.iseval:
                data_t = spec_augment(data_t, **self.text_feat_opt['spec_augment'])
        return (data_v, data_a, data_t), label, fid

    def get_input_dims(self):
        data_v, data_a, data_t = self[0][0]
        nv = 0 if data_v is None else data_v.shape[-1]
        na = 0 if data_a is None else data_a.shape[-1]
        nt = 0 if data_t is None else data_t.shape[-1]
        return nv, na, nt

    def _get_input_layers(self, data):
        if data is None:
            return 0
        elif len(data.shape) == 2:
            return 1
        else:
            return data.shape[0]

    def get_input_layers(self):
        data_v, data_a, data_t = self[0][0]
        nv = self._get_input_layers(data_v)
        na = self._get_input_layers(data_a)
        nt = self._get_input_layers(data_t)
        return nv, na, nt

        nv = 0 if data_v is None else data_v.shape[-1]
        na = 0 if data_a is None else data_a.shape[-1]
        nt = 0 if data_t is None else data_t.shape[-1]
        return nv, na, nt

    def get_output_dims(self):
        labels = np.array(list(self.labels.values()))
        assert len(labels.shape) == 2, 'wrong label format'

        if 'float' in str(labels.dtype):
            return 1, 'regress'

        n_class = []
        for i in range(labels.shape[1]):
            _labels = labels[:, i].flatten()
            n_class.append(np.unique(_labels).size)

        if len(n_class) == 1:
            if n_class[0] == 2:
                return 2, 'multiclass'
            elif n_class[0] > 2:
                return n_class[0], 'multiclass'
            else:
                raise ValueError('label must be either multi-label binary or multi-class')
        else:
            if len(set(n_class)) == 1 and n_class[0] == 2:
                return len(n_class), 'binary'
            else:
                raise ValueError('label must be either multi-label binary or multi-class')

    def get_class_weight(self):
        labels = np.array(list(self.labels.values()))
        assert len(labels.shape) == 2, 'wrong label format'

        w = []
        for i in range(labels.shape[1]):
            _labels = labels[:, i].flatten()
            _label_set, _label_counts = np.unique(_labels, return_counts=True)
            _weight = 1/_label_counts
            _weight /= _weight.sum()/_label_set.size
            w.append(_weight)
        return w


def format_multimodal_data(DataLoaderBatch):
    data_multi, label, uids = zip(*DataLoaderBatch)
    # adjust the same length + create torch.tensor
    data_v, data_a, data_t = zip(*data_multi)
    data_v, data_v_len = pad_and_concat(data_v)
    data_a, data_a_len = pad_and_concat(data_a)
    data_t, data_t_len = pad_and_concat(data_t)
    # create torch.tensor
    label = torch.from_numpy(np.stack(label))
    if label.shape[1] == 1:
        label = label.view(-1)
    return (data_v, data_v_len, data_a, data_a_len, data_t, data_t_len), label, uids


def pad_and_concat(x_list):
    """
    x_list: 2-d/3-d tensor list [nlen, ndim] / [nlay, nlen, ndim]
    """
    if x_list[0] is None:
        return None, None
    else:
        input_shape = len(x_list[0].shape)
        ndim = x_list[0].shape[-1]
        x_len = torch.Tensor([_x.shape[-2] for _x in x_list])
        max_len = int(x_len.max())
        if input_shape == 2:
            padded_x_list = [
                    np.concatenate([_x, np.zeros((max_len-_x.shape[0], ndim))], axis=0)
                    for _x in x_list
                    ]
        else:
            padded_x_list = [
                    np.concatenate([_x, np.zeros((_x.shape[0], max_len-_x.shape[1], ndim))], axis=1)
                    for _x in x_list
                    ]

        x = torch.from_numpy(np.stack(padded_x_list))
        return x.float(), x_len.long()
