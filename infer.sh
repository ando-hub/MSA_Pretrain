#!/bin/bash

gpuid=-1
num_workers=12

dataset=cmumosi
input_modal='videoaudiotext'

feat=pretrained          # pretrained / mmdatasdk / mmsa
label=sentiment_regress

video_data=./data/dataset/$dataset/feat/video/$feat
audio_data=./data/dataset/$dataset/feat/audio/$feat
text_data=./data/dataset/$dataset/feat/text/$feat

test_list="./data/dataset/$dataset/label/$label/test.txt"

modeld='./data/trained/sentiment_regress/cmumosi/pretrained/input_videoaudiotext.model_enc1x128sap4dec1_gate_dec2.train_mb16ep50adam0.0001schedspecaugseed0.feat_layerbestmosi'

config_train='./conf/train/mb16ep50adam0.0001schedspecaugseed0.yaml'
config_feat='./conf/feat/layerbestmosi.yaml'

# main
python scripts/train/train.py \
    --init-model $modeld/model/model.pt \
    --embed-save-dir $modeld/infer/embedding \
    --attn-save-dir $modeld/infer/attention \
    --result-dir $modeld/infer/result \
    --config-train $config_train \
    --config-feat $config_feat \
    --gpu $gpuid \
    --num-workers $num_workers \
    -l $modeld/get_embeddings.log \
    --loglevel debug \
    --audio-data $audio_data \
    --video-data $video_data \
    --text-data $text_data \
    --testset-list $test_list \
    --input-modal $input_modal \
    --test-only \

