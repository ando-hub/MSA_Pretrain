import os
import glob
import argparse
import numpy as np
import pdb
import h5py
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

label_names = np.array(['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear'])


def _parse():
    parser = argparse.ArgumentParser(description='open hdf5 data')
    parser.add_argument('embd', type=str, help='input embedding dir')
    parser.add_argument('label', type=str, help='input label file')
    parser.add_argument('pngd', type=str, help='output png file')
    return parser.parse_args()


def load_label(labelf):
    labdic = {}
    for l in open(labelf):
        fid, labs = l.strip().split(' ', 1)
        labs = np.array([int(v) for v in labs.split()])
        if labs.sum() <= 1:
            index = np.where(labs == 1)
            if index[0].size:
                __label_name = label_names[index][0]
            else:
                __label_name = 'neutral'
            labdic[fid] = __label_name
    return labdic


def _main():
    args = _parse()
    os.makedirs(args.pngd, exist_ok=True)
    
    labdic = load_label(args.label)
    
    video_vecs = []
    audio_vecs = []
    text_vecs = []
    labels = []
    print('load embedding vectors ...')
    for npz in tqdm(sorted(glob.glob(os.path.join(args.embd, '*.npz')))):
        fid = os.path.splitext(os.path.basename(npz))[0]
        if fid in labdic:
            labels.append(labdic[fid])
            src = np.load(npz)
            video_vecs.append(src['emb_video'])
            audio_vecs.append(src['emb_audio'])
            text_vecs.append(src['emb_text'])
    labels = np.stack(labels)
    video_vecs = np.stack(video_vecs)
    audio_vecs = np.stack(audio_vecs)
    text_vecs = np.stack(text_vecs)
    
    print('plot distrib ...')
    plot(video_vecs, labels, os.path.join(args.pngd, 'video_embeddings.png'))
    plot(audio_vecs, labels, os.path.join(args.pngd, 'audio_embeddings.png'))
    plot(text_vecs, labels, os.path.join(args.pngd, 'text_embeddings.png'))


def plot(vecs, labels, png):
    vecs_transformed = TSNE(n_components=2, random_state=0).fit_transform(vecs)
    
    plt.figure()
    label_set = np.unique(labels)
    cmap = plt.get_cmap('jet', label_set.size)
    for i, label in enumerate(label_set):
        color = cmap(i)
        x = vecs_transformed[labels == label]
        plt.scatter(x[:, 0], x[:, 1], color=color, label=label)
    plt.legend()
    plt.savefig(png)
    plt.close()


if __name__ == '__main__':
    _main()
