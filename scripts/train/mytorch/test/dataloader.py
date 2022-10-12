import json
import kaldi_io
import numpy as np
import copy
from pdb import set_trace

# torch v1.3.0+
import torch
from .transform_feat import SpecTransformer

IGNORE_INDEX = -1


def _lab2tensor(l, ohv=False):
    if isinstance(l, int):
        tns = torch.tensor(l).long()
    else:
        tns = torch.tensor(l).float()
    return tns


def create_soft_label(annot, target_class, soft_bias=0.):
    counts = np.zeros(len(target_class), dtype=np.float32)
    for a in annot:
        for i in range(len(a)):
            if a[i] != IGNORE_INDEX:
                counts[a[i]] += 1

    if counts.sum() > 0:
        soft = counts + soft_bias
        soft = soft / soft.sum()
    else:
        soft = None
    return soft


def create_mlee_label(annot, target_class, mlee_thresh=[]):
    if len(mlee_thresh) == 2:
        min_th, max_th = sorted(mlee_thresh)
    elif len(mlee_thresh) == 1:
        min_th = mlee_thresh[0]
        max_th = min_th
    elif len(mlee_thresh) == 0:
        min_th, max_th = None
    else:
        raise ValueError('invalid mlee_thresh')

    counts = np.zeros(len(target_class), dtype=np.float32)
    for a in annot:
        for i in range(len(a)):
            if a[i] != IGNORE_INDEX:
                counts[a[i]] += 1

    n_annots = len(annot)
    annot_ratio = counts/n_annots
    if min_th is not None:
        mlee = np.ones(len(target_class), dtype=np.float32) * IGNORE_INDEX
        mlee[annot_ratio >= max_th] = 1
        mlee[annot_ratio < min_th] = 0
    else:
        mlee = annot_ratio
    return mlee


def is_skip(_in, dtype):
    """
    _in     : label dict {'annotation':[[...], ...], 'maj':0, ...}
    dtype   : available data information
    """
    if _in['maj'] != IGNORE_INDEX:
        in_dtype = 'major'
    elif set([_a for a in _in['annotation'] for _a in a]) - {IGNORE_INDEX}:
        in_dtype = 'minor'
    else:
        in_dtype = 'none'

    if dtype == 'major':
        _skip = True if in_dtype in ['minor', 'none'] else False
    elif dtype == 'minor':
        _skip = True if in_dtype in ['none'] else False
    elif dtype == 'none':
        _skip = False
    elif dtype == 'minor_only':
        _skip = True if in_dtype in ['major', 'none'] else False
    elif dtype == 'minor_none':
        _skip = True if in_dtype in ['major'] else False
    return _skip


def get_class_set(labdic, key):
    vals = []
    for v in labdic.values():
        assert key in v, 'No exist {} in {}'.format(key, v)
        if isinstance(v[key], list):
            vals.extend(v[key])
        else:
            vals.append(v[key])
    return list(set(vals))


def _copy_lists(lists):
    N = [len(l) for l in lists]
    if len(set(N)) == 1:
        return lists
    else:
        if len(set(N)) > 2 or sorted(list(set(N)))[-2] != 1:
            raise ValueError('invalid list length: {}'.format(N))
        Nmax = max(N)
        _lists = []
        for l in lists:
            if len(l) != Nmax:
                l = [copy.deepcopy(l[0]) for n in range(Nmax)]
            _lists.append(l)
        return _lists


def _create_tensors(lab, tgt, use_indiv_labels, slct_worker, class_set):
    tensors = []
    for i, _tgt in enumerate(tgt):
        _tensors = []
        if use_indiv_labels and _tgt in ['maj', 'soft', 'mlee', 'worker']:
            for w, a in zip(lab['worker'], lab['annotation']):
                # skip check
                if w == IGNORE_INDEX:
                    continue
                elif slct_worker and w not in slct_worker:
                    continue

                # get class
                if _tgt == 'maj' or _tgt == 'worker':
                    for _a in a:
                        if _a != IGNORE_INDEX:
                            if _tgt == 'maj':
                                t = _a
                            else:
                                t = w
                            _tensors.append(_lab2tensor(t))
                else:
                    if _tgt == 'soft':
                        t = create_soft_label(a, class_set[i], soft_bias)
                    elif _tgt == 'mlee':
                        t = create_mlee_label(a, class_set[i], mlee_thresh)
                    _tensors.append(_lab2tensor(t))
        else:
            if _tgt == 'soft':
                t = create_soft_label(
                        [_a for a in lab['annotation'] for _a in a],
                        class_set[i],
                        soft_bias,
                        )
            elif _tgt == 'mlee':
                t = create_mlee_label(
                        [_a for a in lab['annotation'] for _a in a],
                        class_set[i],
                        soft_bias,
                        )
            elif _tgt == 'worker':
                # raise ValueError('worker is invalid when use_indiv_labels == False')
                t = lab[_tgt][0]    # slice first indices
            else:
                t = lab[_tgt]
            _tensors.append(_lab2tensor(t))
        tensors.append(_tensors)
    if not tensors:
        return (None, )
    else:
        return list(zip(*_copy_lists(tensors)))


class MyDataset(torch.utils.data.Dataset):
    def __init__(
            self, scps, labfs, tgt=['maj'], aux=[],
            transform=None, soft_bias=0., mlee_thresh=[0.01],
            dtype='major',
            use_indiv_labels=False,
            rm_invalid_worker=False,
            slct_worker=[],
            ):
        self.transform = transform
        self.tgt = tgt
        self.aux = aux

        self.label = []     # reference labels
        self.data = []      # input features (ark path/position)
        self.data_aux = []  # input auxiliary labels
        self.uid = []       # utterance id

        self.target_class = []
        self.garbage_class = None

        # load label
        labdic = {}
        for labf in labfs:
            with open(labf) as fp:
                labdic.update(json.load(fp))

        # prepare target class / relabeling
        tgt_class_set = []
        for _tgt in tgt:
            tgt_key = 'maj' if _tgt in ['soft', 'mlee'] else _tgt
            _class = []
            for v in labdic.values():
                if isinstance(v[tgt_key], list):
                    _class.extend(v[tgt_key])
                else:
                    _class.append(v[tgt_key])
            tgt_class_set.append(set(_class)-{IGNORE_INDEX})

        aux_class_set = []
        for _aux in aux:
            _class = []
            for v in labdic.values():
                if isinstance(v[_aux], list):
                    _class.extend(v[_aux])
                else:
                    _class.append(v[_aux])
            aux_class_set.append(set(_class)-{IGNORE_INDEX})

        # load scps
        for scp in scps:
            for l in open(scp):
                uid, pos = l.strip().split()
                # skip invalid data
                if uid not in labdic:
                    continue
                elif is_skip(labdic[uid], dtype):
                    continue
                elif rm_invalid_worker and IGNORE_INDEX in labdic[uid]['worker']:
                    continue

                _lab = labdic[uid]
                t = _create_tensors(
                        _lab, tgt,
                        use_indiv_labels, slct_worker, tgt_class_set)
                a = _create_tensors(
                        _lab, aux,
                        use_indiv_labels, slct_worker, aux_class_set)

                if t:
                    t, pos, a, uid = _copy_lists([t, [pos], a, [uid]])
                    self.label.extend(t)
                    self.data.extend(pos)
                    self.data_aux.extend(a)
                    self.uid.extend(uid)

        self.nout = tuple([max(s)+1 for s in tgt_class_set])
        self.naux = tuple([max(s)+1 for s in aux_class_set])
        self.aux_len = len(self.naux)
        self.n_data = len(self.data)
        _x = self.__getitem__(0)[0]
        self.nin = _x.shape[1]

        """
        # calculate worker-wise weight & overwrite (TBM: avoid hard-code)
        self.weight_worker = None
        if self.tgt == ['maj'] and self.aux and self.aux[0] == 'worker':
            # calculate worker-wise weight
            t = torch.stack([l[0] for l in self.label])
            w = torch.stack([a[0] for a in self.data_aux])
            weight_worker = []
            for i in range(self.naux[0]):
                _weight = self.__calc_cw_1d(t[w==i], self.nout[0])
                weight_worker.append(_weight)
            self.weight_worker = torch.stack(weight_worker)

            # overwrite ohv(int) to weighted ohv(vector)
            label = []
            for idx in range(self.n_data):
                ohv = torch.eye(self.nout[0])[self.label[idx][0]] * self.weight_worker[w[idx]]
                label.append((ohv, ))
            self.label = label
            print('overwrite label file ...')
        """

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        # load input features from path
        x = kaldi_io.read_mat(self.data[idx])
        if self.transform:
            x = self.transform(x)
        out_data = torch.from_numpy(x).float()

        # convert auxiliary features to (concatenated) one-hot vectors
        if self.naux:
            ohvs = [torch.eye(self.naux[i])[self.data_aux[idx][i]]
                    for i in range(self.aux_len)]
            out_aux_data = torch.cat(ohvs, dim=0)
        else:
            out_aux_data = None

        # load label, uid
        out_label = self.label[idx]
        out_uid = self.uid[idx]
        return out_data, out_aux_data, out_label, out_uid

    def get_class_weight(self):
        class_weight = []
        for i, _tgt in enumerate(self.tgt):
            _t = torch.stack([v[i] for v in self.label], dim=0)
            if _tgt == 'mlee':
                w = self.__calc_cw_multilab(_t)
            elif len(_t.size()) == 1:
                w = self.__calc_cw_1d(_t, self.nout[i])
            elif len(_t.size()) == 2:
                w = self.__calc_cw_2d(_t)
            else:
                raise NotImplementedError
            class_weight.append(w)
        return class_weight

    def get_class_weight_worker(self, worker_weight=False):
        if self.tgt == ['maj'] and self.aux and self.aux[0] == 'worker':
            # calculate worker-wise weight
            t = torch.stack([l[0] for l in self.label])
            w = torch.stack([a[0] for a in self.data_aux])
            N = w.shape[0]

            weight_worker = []
            for i in range(self.naux[0]):
                _weight = self.__calc_cw_1d(t[w==i], self.nout[0])
                if worker_weight:
                    _weight /= (w==i).sum()
                weight_worker.append(_weight)
            weight = torch.stack(weight_worker)
            weight = weight * (self.nout[0]*self.naux[0]/weight.sum())  # class/worker avg = 1.
            return weight
        else:
            raise ValueError('--task maj && --aux worker only')

    def set_class_weight_worker(self, weight_worker):
        # overwrite ohv(int) to weighted ohv(vector)
        label = []
        for idx in range(self.n_data):
            ohv = torch.eye(self.nout[0])[self.label[idx][0]] * weight_worker[self.data_aux[idx][0]]
            label.append((ohv, ))
        self.label = label
        print('overwrite label file ...')

    def get_dims(self):
        return self.nin, sum(self.naux), self.nout

    def get_workers(self):
        return range(self.naux[self.aux.index('worker')])

    def get_uniq_uttrs(self):
        return list(set(self.uid))

    def get_garbage_class(self):
        return self.garbage_class

    def __calc_cw_1d(self, t, ndim):
        w = torch.zeros(ndim).float()
        for i, c in enumerate(range(ndim)):
            freq = t[t==c].size(0)
            if freq > 0:
                w[i] = 1/freq
        w /= w.sum()/ndim            # avg weight = 1 (actually not required)
        return w

    def __calc_cw_2d(self, t):
        w = 1./t.mean(dim=0)
        w /= w.sum()/len(self.target_class)
        return w

    def __calc_cw_multilab(self, t):
        freq = t.mean(dim=0)
        if (freq==0).any():
            raise ValueError('all labels may be 0 in any dimensions')
        if (freq==1).any():
            raise ValueError('all labels may be 1 in any dimensions')

        w0 = 0.5/(1-freq)
        w1 = 0.5/freq
        return torch.stack([w0, w1], dim=0)


class MyDataLoaderFormatter(object):
    def __init__(self):
        pass

    def __call__(self, DataLoaderBatch):
        x_list, aux_list, t_list, uid_list = zip(*DataLoaderBatch)
        x, x_len = pad_and_concat(x_list)
        if aux_list[0] is None:
            aux = None
        else:
            aux = torch.stack(aux_list, dim=0)
        t = concat_multi_target(t_list)
        x, x_len, aux, t, uid_list = descend_sort(x, x_len, aux, t, uid_list)
        return x, x_len, aux, t, uid_list


def pad_and_concat(x_list):
    """
    x_list: 2-d tensor list [nlen x ndim]
    """
    ndim = x_list[0].shape[1]
    x_len = torch.Tensor([_x.shape[0] for _x in x_list])
    max_len = int(x_len.max())

    padded_x_list = [
            torch.cat([_x, torch.zeros((max_len-_x.shape[0], ndim))], dim=0)
            for _x in x_list
            ]
    x = torch.stack(padded_x_list, dim=0)
    return x.float(), x_len.long()


def concat_multi_target(t_list):
    """
    t_list: list of target tuple e.g. [([0.2, 0.8], 0), ([0.6, 0.4], 1), ...]
    """
    return [torch.stack(_t, dim=0) for _t in zip(*t_list)]

def descend_sort(x, x_len, aux, t, uid):
    """
    x       : Tensor [nbat x nlen x ndim]
    x_len   : Tensor [nbat]
    aux     : Tensor [nbat x ndim]
    t       : list of Tensor (Tensor [nbat], Tensor [nbat], ...)
    uid     : list of string
    """
    x_len, perm_idx = x_len.sort(0, descending=True)
    x = x[perm_idx]
    aux = aux[perm_idx] if aux is not None else None
    t = [_t[perm_idx] for _t in t]
    uid = [uid[i] for i in perm_idx]
    return x, x_len, aux, t, uid


def _utest():
    scp = ['/data1/191023_emo_journal/201_packfeat/00_spectrogram/novad/IEMOCAP-impro/000/tst/feats.scp']
    lab = ['/nfs/lnx/work0023/ando/200306_emo_Listener/100_lab/00_lab/IEMOCAP.json']
    #lab = ['/nfs/lnx/work0023/ando/200306_emo_Listener/100_lab/00_lab/IEMOCAP.exc2hap.json']
    tgt = ['maj']
    aux = ['worker']
    #aux = []
    worker= []

    normf = '/data1/191023_emo_journal/203_normalize/00_norm_uniform/novad/IEMOCAP-impro/000/norm.txt'
    transform = SpecTransformer(normf=normf, iseval=True)
    dataset = MyDataset(
            scp, lab, tgt, aux,
            transform=transform,
            dtype='major', device='cpu', primal_label_only=False,
            use_garbage=False, use_indiv_labels=True, rm_invalid_worker=True,
            slct_worker=worker,
            )

    from torch.utils.data import DataLoader
    formatter = MyDataLoaderFormatter()
    loader = DataLoader(dataset, collate_fn=formatter, batch_size=8, shuffle=True)

    for d in loader:
        # set_trace()
        pass

    set_trace()


if __name__=='__main__':
    _utest()
