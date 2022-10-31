# Multimodal Sentiment Analysis

Official pytorch implementation of [On the Use of Modalty-Specific Large-Scale Pre-Trained Encoders for Multimodal Sentiment Analysis (SLT2022)](https://arxiv.org/abs/2210.15937).

```
@inproceedings{ando2022MSA_Pretrain,
title={On the Use of Modalty-Specific Large-Scale Pre-Trained Encoders for Multimodal Sentiment Analysis},
    author={Atsush Ando, Ryo Masumura, Akihiko Takashima, Satoshi Suzuki, Naoki Makishima, Keita Suzuki, Takafumi Moriya, Takanori Ashihara, Hiroshi Sato},
    booktitle={Proceedings of the 2022 IEEE Spoken Language Technology Workshop (SLT 2022)},
    pages={to appear},
    year={2022}
  }
```

## License
Please read `LICENSE`  before using scripts.


## Requirements
- Python >= 3.9
- pytorch >= 1.12
- CUDA >= 10.2

## Usage

### 1. Install dependencies
```
pip install -r requirements
```

### 2. Save pretrained WavLM model
```
(download WavLM Large model from https://github.com/microsoft/unilm/tree/master/wavlm)
cp ./WavLM-Large.pt conf/pretrained_enc

```

### 3. Run setup scripts

Please edit `[cmumosi|cmumosei] dataset path` in the scripts before running if you have CMU-MOSI/CMU-MOSEI corpora.
```
./setup_cmumosi.sh
./setup_cmumosei.sh
```

### 4. Run training script
```
./train.sh
```
- Edit the following variables to try different setups
    - dataset: `cmumosi` or `cmumosei`
    - input\_modal: `video`, `audio`, `text`, or `videoaudiotext`
    - feat: `mmdatasdk_noalign`(Conv in paper), `mmsa_noalign`(Conv-BERT), or `pretrained`(Enc. \*)
    - config\_models
    - config\_feats:
        - `layerXX`: outputs of (XX+1)-th encoder layer (layer23 = Enc. output)
        - `layerbest`: combination of best intermediate encoder layers (Enc. mid-best)
        - `layerall`: weighted sum of the outputs of intermediate encoder layers (Enc. weighted)
- You can see the performance in `$rsltd/result.tst.txt`
    - [task: 2/3/5/7/nz2] mean 2(neg/nonneg)/3/5/7/2(neg/pos)-class classification performances. WA, UA, MF1, WF1 is weighted accuracy, unweighted accuracy, macro F1, and weighted F1, respectively
    - [task: reg\_regress] is regression performance
- ***The results may slightly different from those in the paper due to cuda nondeterministic behavior*** (see: https://pytorch.org/docs/stable/notes/randomness.html). Results by this repos are:


|                      |MOSI-MAE |MOSI-Corr|MOSEI-MAE|MOSEI-Corr|
|:---------------------|--------:|--------:|--------:|---------:|
|mmdatasdk (Conv)      |.934     |.667     |.598     |.684      |
|mmsa (Conv-BERT)      |.889     |.691     |.542     |.748      |
|P/T Enc. output       |.844     |.716     |.521     |.772      |
|P/T Enc. mid-best     |.812     |.747     |.507     |.789      |
|P/T Enc. weighted     |.833     |.751     |.511     |.785      |


## Contact
mail: atsushi.ando.hd@hco.ntt.co.jp

