# Multimodal Sentiment Analysis

Official pytorch implementation of [On the Use of Modalty-Specific Large-Scale Pre-Trained Encoders for Multimodal Sentiment Analysis (SLT2022)](https://arxiv.org/abs/xxxx.xxxxx).

## License
Please read `LICENSE`  before using scripts.

## Usage

1. Install dependencies
```
pip install -r requirements
```

2. Save pretrained WavLM model
```
(download WavLM Large model from https://github.com/microsoft/unilm/tree/master/wavlm)
cp ./WavLM-Large.pt conf/pretrained_enc

```

3. Run setup scripts

Please edit `[cmumosi|cmumosei] dataset path` in the scripts before running if you have CMU-MOSI/CMU-MOSEI corpora.
```
./setup_cmumosi.sh
./setup_cmumosei.sh
```

4. Run training script
```
./train.sh
```

## Paper
Please cite the following paper if you find our work is useful in your research:

```
@inproceedings{ando2022MSA_UEGD,
title={On the Use of Modalty-Specific Large-Scale Pre-Trained Encoders for Multimodal Sentiment Analysis},
    author={Atsush Ando, Ryo Masumura, Akihiko Takashima, Satoshi Suzuki, Naoki Makishima, Keita Suzuki, Takafumi Moriya, Takanori Ashihara, Hiroshi Sato},
    booktitle={Proceedings of the 2022 IEEE Spoken Language Technology Workshop (SLT 2022)},
    pages={to appear},
    year={2022}
  }
```

