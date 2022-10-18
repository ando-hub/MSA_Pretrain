# TBM: replace bert-base/bert-large to those based on huggingface, same as gpt2/t5
import os
import glob
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model, T5Tokenizer, T5EncoderModel

torch.use_deterministic_algorithms(True)
repos_dict = {
        'bert-base': 'bert-base-uncased',
        'bert-large': 'bert-large-uncased',
        'gpt2': 'gpt2',
        't5-base': 'google/t5-v1_1-base',
        't5-large': 'google/t5-v1_1-large',
        }


def _parse():
    parser = argparse.ArgumentParser(description='Create text hdf5 file')
    parser.add_argument('ind', type=str, help='input text rootdir')
    parser.add_argument('outd', type=str, help='output npy dir')
    parser.add_argument('--token-outd', type=str, help='output tokenized text dir')
    parser.add_argument('--encoder-type',
                        choices=repos_dict.keys(),
                        default='bert-large',
                        help='model type: bert-[base|large], gpt2, t5-[base|large]')
    parser.add_argument('--gpuid', type=int, default=-1,
                        help='gpu id (run cpu if gpuid < 0)')
    parser.add_argument('--get-layer-results', action='store_true', default=False,
                        help='get results of the all encoder layers')
    parser.add_argument('--batchsize', type=int, default=1,
                        help='proc #batchsize files per decode')
    return parser.parse_args()


class TextEmbeddingExtractor():
    def __init__(self, encoder_type='bert-large', get_layer_results=False, device='cpu'):
        self.device = device
        self.get_layer_results = get_layer_results
        self.encoder_type = encoder_type

        if encoder_type in ['bert-base', 'bert-large']:
            self.tokenizer = BertTokenizer.from_pretrained(repos_dict[encoder_type])
            self.model = BertModel.from_pretrained(repos_dict[encoder_type])
        elif encoder_type == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(repos_dict[encoder_type])
            self.model = GPT2Model.from_pretrained(repos_dict[encoder_type])
        elif encoder_type in ['t5-base', 't5-large']:
            self.tokenizer = T5Tokenizer.from_pretrained(repos_dict[encoder_type])
            self.model = T5EncoderModel.from_pretrained(repos_dict[encoder_type])
        else:
            raise ValueError('invalid encoder_type: {}'.format(encoder_type))
        self.model.eval()
        self.model.to(device)

    def tokenize(self, txt, save_path=None):
        if self.encoder_type.startswith('bert'):
            txt = "[CLS] " + txt + " [SEP]"
        tokens = self.tokenizer.tokenize(txt)
        if save_path:
            with open(save_path, 'w') as fp:
                fp.write(' '.join(tokens))
        return np.array(self.tokenizer.convert_tokens_to_ids(tokens), dtype=np.int64)

    def decode(self, x, x_len=None):
        segments = torch.zeros(x.shape, dtype=torch.long, device=self.device)
        x = torch.tensor(x).to(device=self.device)
        if x_len is None:
            input_mask = None
        else:
            x_len = torch.tensor(x_len).to(self.device)
            input_mask = torch.arange(x.shape[1])[None, :].to(self.device) < x_len[:, None]

        with torch.no_grad():
            if self.encoder_type.startswith('bert'):
                layer_results, _ = self.model(x, segments, attention_mask=input_mask)
            else:
                outputs = self.model(input_ids=x, attention_mask=input_mask, output_hidden_states=True)
                layer_results = outputs['hidden_states']

        # reshape: [n_bat, n_lay, n_len, n_dim]
        if self.get_layer_results:
            y = torch.stack(layer_results).transpose(0, 1)
        else:
            y = layer_results[-1].unsqueeze(1)

        # return
        if x_len is None:
            return y.cpu().detach().numpy()
        else:
            ret = []
            for i in range(y.shape[0]):
                valid_indices = input_mask[i].nonzero().view(-1)
                yi = y[i].index_select(1, valid_indices)
                ret.append(yi.cpu().detach().numpy())
            return ret

    def extract_fromfiles(self, infs, outd, tokend=None):
        # create tensor, length
        data = []
        length = []
        for f in infs:
            with open(f) as fp:
                txt = fp.read().strip()
            tokenf = os.path.join(tokend, os.path.basename(f)) if tokend else None
            indices = self.tokenize(txt, tokenf)
            data.append(indices)
            length.append(len(indices))

        # zero padding
        max_length = max(length)
        data = [np.concatenate((d, np.zeros(max_length-len(d), dtype=d.dtype))) for d in data]
        data = np.stack(data, axis=0)

        # decode
        y = self.decode(data, length)

        # save results to .npy
        for f, emb in zip(infs, y):
            clip_id = os.path.splitext(os.path.basename(f))[0]
            outf = os.path.join(outd, clip_id+'.npy')
            np.save(outf, emb)


def _main():
    args = _parse()
    os.makedirs(args.outd, exist_ok=True)
    if args.token_outd:
        os.makedirs(args.token_outd, exist_ok=True)
    device = 'cuda:{}'.format(args.gpuid) if args.gpuid >= 0 else 'cpu'

    print('prepare extractor')
    extractor = TextEmbeddingExtractor(
            encoder_type=args.encoder_type,
            get_layer_results=args.get_layer_results,
            device=device
            )

    # get unproceeded files for resume
    infs = []
    for f in glob.glob(os.path.join(args.ind, '*.txt')):
        clip_id = os.path.splitext(os.path.basename(f))[0]
        outf = os.path.join(args.outd, clip_id+'.npy')
        if not os.path.exists(outf):
            infs.append(f)

    print('start feature extraction')
    for fs in tqdm([infs[i:i+args.batchsize] for i in range(0, len(infs), args.batchsize)]):
        try:
            extractor.extract_fromfiles(fs, args.outd, args.token_outd)
        except RuntimeError as e:
            print(e)


if __name__ == '__main__':
    _main()
