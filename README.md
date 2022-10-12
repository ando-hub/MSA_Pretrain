# ON THE USE OF MODALITY-SPECIFIC LARGE-SCALE PRE-TRAINED ENCODERS FOR MULTIMODAL SENTIMENT ANALYSIS

Official pytorch implementation of [On the Use of Modalty-Specific Large-Scale Pre-Trained Encoders for Multimodal Sentiment Analysis (SLT2022)](https://arxiv.org/abs/xxxx.xxxxx). 
Please read `LICENSE` before using scripts.

## Usage

1. Install dependencies
```
pip install -r requirements
```
2. Run setup scripts
Please edit the scripts to set CMU-MOSI/CMU-MOSEI root path before running if you have these corpora.
```
./setup_cmumosi.sh
./setup_cmumosei.sh
```

3. Run training scripts
```
./train.sh
```


## Paper
Please cite our paper if you find our work useful for your research:

```
@inproceedings{ando2022MSA_UEGD,
title={On the Use of Modalty-Specific Large-Scale Pre-Trained Encoders for Multimodal Sentiment Analysis},
    author={Atsush Ando, Ryo Masumura, Akihiko Takashima, Satoshi Suzuki, Naoki Makishima, Keita Suzuki, Takafumi Moriya, Takanori Ashihara, Hiroshi Sato},
    booktitle={Proceedings of the 2022 IEEE Spoken Language Technology Workshop (SLT 2022)},
    pages={to appear},
    year={2022}
  }
```

