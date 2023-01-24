import os
import glob
import argparse
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import soundfile as sf
# add WavLM library
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'extlib/unilm/wavlm'))
from WavLM import WavLM, WavLMConfig

torch.use_deterministic_algorithms(True)


def _parse():
    parser = argparse.ArgumentParser(description='Extract audio features and save in hdf5')
    parser.add_argument('ind', type=str, help='input audio rootdir')
    parser.add_argument('outd', type=str, help='output npy dir')
    parser.add_argument('--encoder-type', type=str,
                        default='wavlm',
                        help='model type: wavlm, [wav2vec2|hubert]_[base|large]')
    parser.add_argument('--wavlm-model-path', type=str, help='wavlm model path')
    parser.add_argument('-r', type=int, default=1,
                        help='subsampling rate of encoder output (Note: WavLM out is 50fps)')
    parser.add_argument('--gpuid', type=int, default=-1,
                        help='gpu id (run cpu if gpuid < 0)')
    parser.add_argument('--get-layer-results', action='store_true', default=False,
                        help='get results of the all encoder layers')
    parser.add_argument('--batchsize', type=int, default=1,
                        help='proc #batchsize .wav files per decode')
    return parser.parse_args()


class AudioEmbeddingExtractor():
    def __init__(self, encoder_type, wavlm_model_path, resamp=1, get_layer_results=False, device='cpu'):
        self.device = device
        self.get_layer_results = get_layer_results
        self.encoder_type = encoder_type
        self.resamp = resamp

        if encoder_type == 'wavlm':
            if not os.path.exists(wavlm_model_path):
                raise ValueError(
                        'invalid --wavlm-model-path: {}\nVisit https://github.com/microsoft/unilm/tree/master/wavlm, download and save WavLM-Large model'.format(wavlm_model_path)
                        )
            checkpoint = torch.load(wavlm_model_path)
            self.cfg = WavLMConfig(checkpoint['cfg'])
            self.model = WavLM(self.cfg)
            self.model.load_state_dict(checkpoint['model'])
        else:
            bundle = getattr(torchaudio.pipelines, encoder_type.upper())
            self.model = bundle.get_model()
        self.model.eval()
        self.model.to(device)

    def extract(self, data, length=None):
        assert isinstance(data, np.ndarray), 'numpy ndarray only'
        assert data.dtype == np.float64, 'normalized audio (np.float64) only'

        # reshape tensor
        if len(data.shape) == 1:
            x = torch.tensor(data).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            padding_mask = None
        elif len(data.shape) == 2:
            x = torch.tensor(data).to(device=self.device, dtype=torch.float32)
            padding_mask = None
            if length is not None:
                length = torch.tensor(length).to(self.device)
                padding_mask = torch.arange(x.shape[1])[None, :].to(self.device) >= length[:, None]

        if self.encoder_type == 'wavlm':
            # decode
            with torch.no_grad():
                (y, layer_results), ret_mask = self.model.extract_features(
                        x,
                        padding_mask=padding_mask,
                        output_layer=self.model.cfg.encoder_layers,
                        ret_layer_results=True
                        )
            # reshape: [n_bat, n_lay, n_len, n_dim]
            if self.get_layer_results:
                # skip 0-th layer of layer_results since it is CNN output and not the outputs of transformer blocks
                # see L591-592 in unilm/wavlm/WavLM.py
                y = torch.cat([x.transpose(0, 1).unsqueeze(1) for x, _ in layer_results[1:]], dim=1)
            else:
                y = y.unsqueeze(1)
            # return
            if padding_mask is None:
                return y.cpu().detach().numpy()
            else:
                ret = []
                for i in range(y.shape[0]):
                    valid_indices = (~ret_mask[i]).nonzero().view(-1)
                    yi = y[i].index_select(1, valid_indices)
                    ret.append(yi.cpu().detach().numpy())
                return ret
        else:
            # decode
            with torch.inference_mode():
                layer_results, ret_length = self.model.extract_features(x, length)
            # reshape
            if self.get_layer_results:
                y = torch.stack(layer_results).transpose(0, 1)
            else:
                y = layer_results[-1].unsqueeze(1)
            # return
            if padding_mask is None:
                return y.cpu().detach().numpy()
            else:
                ret = []
                for i in range(y.shape[0]):
                    yi = y[i][:, :ret_length[i]]
                    ret.append(yi.cpu().detach().numpy())
                return ret

    def extract_fromfiles(self, infs, outd=None):
        # create tensor, length
        data = []
        length = []
        for f in infs:
            x, _ = sf.read(f)
            if len(x.shape) > 1:
                x = x[:, 0]
            data.append(x)
            length.append(len(x))

        # zero padding
        max_length = max(length)
        data = [np.concatenate((d, np.zeros(max_length-len(d), dtype=d.dtype))) for d in data]
        data = np.stack(data, axis=0)

        # decode
        y = self.extract(data, length)

        # save results to .npy
        for f, emb in zip(infs, y):
            clip_id = os.path.splitext(os.path.basename(f))[0]
            outf = os.path.join(outd, clip_id+'.npy')
            np.save(outf, emb[:, ::self.resamp])


def _main():
    args = _parse()
    os.makedirs(args.outd, exist_ok=True)
    device = 'cuda:{}'.format(args.gpuid) if args.gpuid >= 0 else 'cpu'

    print('prepare extractor')
    extractor = AudioEmbeddingExtractor(
            args.encoder_type,
            args.wavlm_model_path,
            resamp=args.r,
            get_layer_results=args.get_layer_results,
            device=device
            )

    # get unproceeded files for resume
    infs = []
    for f in glob.glob(os.path.join(args.ind, '*.wav')):
        clip_id = os.path.splitext(os.path.basename(f))[0]
        outf = os.path.join(args.outd, clip_id+'.npy')
        if not os.path.exists(outf):
            infs.append(f)

    print('start feature extraction')
    for fs in tqdm([infs[i:i+args.batchsize] for i in range(0, len(infs), args.batchsize)]):
        try:
            extractor.extract_fromfiles(fs, args.outd)
        except RuntimeError as e:
            print(e)


if __name__ == '__main__':
    _main()
