#!/bin/bash

gpuid=3
num_workers=12

dataset=cmumosi                    # cmumosi / cmumosei
input_modals='videoaudiotext'

feat=mmsa_noalign                     # pretrained / mmdatasdk_noalign / mmsa_noalign
label=sentiment_regress

video_data=./data/dataset/$dataset/feat/video/$feat
audio_data=./data/dataset/$dataset/feat/audio/$feat
text_data=./data/dataset/$dataset/feat/text/$feat
train_list=./data/dataset/$dataset/label/$label/train.txt
valid_list=./data/dataset/$dataset/label/$label/valid.txt
test_list=./data/dataset/$dataset/label/$label/test.txt

out_base="./data/trained/$label/$dataset/$feat"
out_merge_base="./data/trained_merge/$label/$dataset/$feat"

config_models='
./conf/model/enc1x128sap4dec1_gate_dec2.yaml
'
#./conf/model/enc1sap4dec1_gate_dec2.yaml
#./conf/model/enc1x128sap4dec1_gate_dec2.yaml

config_trains='
./conf/train/mb16ep50adam0.0001schedspecaugseed0.yaml
./conf/train/mb16ep50adam0.0001schedspecaugseed1.yaml
./conf/train/mb16ep50adam0.0001schedspecaugseed2.yaml
./conf/train/mb16ep50adam0.0001schedspecaugseed3.yaml
./conf/train/mb16ep50adam0.0001schedspecaugseed4.yaml
'

config_feats="
./conf/feat/layer23.yaml
"
#./conf/feat/layerall.yaml
#./conf/feat/layerbestmosi.yaml
#./conf/feat/layerbestmosei.yaml
#./conf/feat/layer23.yaml
#./conf/feat/layer22.yaml
#./conf/feat/layer21.yaml
#./conf/feat/layer20.yaml
#./conf/feat/layer19.yaml
#./conf/feat/layer18.yaml
#./conf/feat/layer17.yaml
#./conf/feat/layer16.yaml
#./conf/feat/layer15.yaml
#./conf/feat/layer14.yaml
#./conf/feat/layer13.yaml
#./conf/feat/layer12.yaml
#./conf/feat/layer11.yaml
#./conf/feat/layer10.yaml
#./conf/feat/layer9.yaml
#./conf/feat/layer8.yaml
#./conf/feat/layer7.yaml
#./conf/feat/layer6.yaml
#./conf/feat/layer5.yaml
#./conf/feat/layer4.yaml
#./conf/feat/layer3.yaml
#./conf/feat/layer2.yaml
#./conf/feat/layer1.yaml
#./conf/feat/layer0.yaml

# main
for config_model in $config_models ; do
    for config_train in $config_trains ; do
        for config_feat in $config_feats ; do
            for input_modal in $input_modals ; do

                outd="$out_base/input_${input_modal}.model_`basename $config_model .yaml`.train_`basename $config_train .yaml`.feat_`basename $config_feat .yaml`"

                modeld="$outd/model"
                rsltd="$outd/result"
                logf="$outd/train.log"
                
                opt=""
                if [ -e $modeld/model.pt ]; then
                    # resume
                    opt="$opt --init-model-dir $modeld"
                fi

                python scripts/train/train.py \
                    --model-save-dir $modeld \
                    --result-dir $rsltd \
                    -l $logf \
                    --loglevel debug \
                    --gpu $gpuid \
                    --num-workers $num_workers \
                    --config-train $config_train \
                    --config-model $config_model \
                    --config-feat $config_feat \
                    --audio-data $audio_data \
                    --video-data $video_data \
                    --text-data $text_data \
                    --trainset-list $train_list \
                    --validset-list $valid_list \
                    --testset-list $test_list \
                    --input-modal $input_modal \
                    $opt
            done
        done
    done
done

# evaluate average scores of trials (MAE, Corr, Acc, F1)
python scripts/util/merge_score.py $out_base $out_merge_base
