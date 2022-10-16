#!/bin/bash

stage=0
stop_stage=0

gpuid=-1                # Use ${gpuid}-th GPU in feature extraction if $gpuid >= 0
extfeat_batchsize=4     # proc every $batchsize files in audio/text feature extraction

# cmumosi dataset path
cmumosi_root='./data/src/CMU-MOSI/Raw'
cmumosi_label_csd='./data/src/CMU-MOSI/mmdatasdk/CMU-MOSI/labels/CMU_MOSI_Opinion_Labels.csd'
cmumosi_feat_mmdatasdk='./data/src/CMU-MOSI/processed_data/cmu-mosi/seq_length_50/mosi_data_noalign.pkl'
cmumosi_feat_mmsa='./data/src/CMU-MOSI/MMSA/Processed/unaligned_50.pkl'

output_dir='./data/dataset/cmumosi'
label_format='sentiment_regress'    # sentiment_regress, sentiment_class, or emo_class

# pre-trained encoder setup
video_encoder='CLIP'
audio_encoder='WavLM'
text_encoder='BERT_Large'


if [ $stage -le 0 ]; then
    # stage 0: download datasets
    # CMU-MOSI raw dataset
    if [ ! -d $cmumosi_root ]; then
        echo "Download CMU-MOSI raw dataset ..."
        mkdir -p $cmumosi_root
        wget -P $cmumosi_root/../ http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSI.zip
        unzip $cmumosi_root/../CMU_MOSI.zip -d $cmumosi_root/../
    fi
 
    # CMU-MOSI labels
    if [ ! -f $cmumosi_label_csd ]; then
        echo "Download CMU-MOSI label ..."
        if [ ! -d ${cmumosi_label_csd%/*} ]; then
            mkdir -p ${cmumosi_label_csd%/*}
        fi
        wget -P ${cmumosi_label_csd%/*} http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/labels/CMU_MOSI_Opinion_Labels.csd
    fi

    # CMU-MOSI features (mmdatasdk)
    if [ ! -f $cmumosi_feat_mmdatasdk ]; then
        echo "Download CMU-MOSI feat (mmdatasdk) ..."
        if [ ! -d ${cmumosi_feat_mmdatasdk%/*} ]; then
            mkdir -p ${cmumosi_feat_mmdatasdk%/*}
        fi
        wget -P ${cmumosi_feat_mmdatasdk%/*} http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosi/seq_length_50/mosi_data_noalign.pkl
    fi

    # CMU-MOSI features (mmsa)
    if [ ! -f $cmumosi_feat_mmsa ]; then
        if [ ! -d ${cmumosi_feat_mmsa%/*} ]; then
            mkdir -p ${cmumosi_feat_mmsa%/*}
        fi
        echo "No exist CMU-MOSI feat (MMSA) !"
        echo "Please access 'https://github.com/thuiar/MMSA', download 'MOSI/Processed/unaligned_50.pkl' in 2.Datasets, and save it to ${cmumosi_feat_mmsa%/*}"
        exit 1
    fi
fi

if [ $stage -le 1 ]; then
    # stage 1: get utterance-level video/audio/text files
    
    # link utteance-level files
    python scripts/preproc/mosi_link_data.py \
        $cmumosi_root/Video/Segmented \
        $output_dir/video/original 
    python scripts/preproc/mosi_link_data.py \
        $cmumosi_root/Audio/WAV_16000/Segmented \
        $output_dir/audio/original 

    # get utterance-level text files from entire transcriptions 
    python scripts/preproc/mosi_segment_text.py \
        $cmumosi_root/Transcript/Segmented \
        $output_dir/text/original
fi

if [ $stage -le 2 ]; then
    # stage 2: create utterance-level labels
    python scripts/preproc/mosi_create_label.py \
        --label-format $label_format \
        $cmumosi_label_csd \
        $output_dir/label/$label_format
fi

if [ $stage -le 3 ]; then
    # 3. frame resampling, face detection and feature extraction from video
    python scripts/extfeat/extract_video_embedding.py \
        $output_dir/video/original \
        $output_dir/feat/video/$video_encoder \
        --fps 3 \
        --face-detect-method facenet \
        --extfeat-method $video_encoder \
        --face-outd $output_dir/video/face \
        --image-outd $output_dir/video/image \
        --gpuid $gpuid \
        --get-layer-results
        --face-size 256 \

fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    # 4. feature extraction from audio
    python scripts/extfeat/extract_audio_embedding.py \
        $output_dir/audio/original \
        $output_dir/feat/audio/tmp_$audio_encoder \
        --gpuid $gpuid \
        --model $audio_encoder \
        --get-layer-results \
        --batchsize $batchsize

    # resample audio (20ms frame = 50fps -> 5fps)
    # TODO: implement resample func in extract_audio_embedding.py
    python scripts/extfeat/resamp_npy.py \
        $output_dir/feat/audio/tmp_$audio_encoder \
        $output_dir/feat/audio/$audio_encoder \
        -r 10
    
    rm -rf $output_dir/feat/audio/tmp_$audio_encoder 
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    # 5. tokenize, indexing and feature extraction from text
    python scripts/extfeat/extract_text_embedding.py \
        $output_dir/text/original \
        $output_dir/feat/text/$text_encoder \
        --gpuid $gpuid \
        --model-type $text_encoder \
        --token-outd $output_dir/text/token_${text_encoder} \
        --get-layer-results \
        --batchsize $extfeat_batchsize
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    # 6. get conventional features for comparison
    # mmdatasdk (CMU-MOSI official features): FACET 35dim, COVEREP 74dim, GloVe 300dim
    python scripts/preproc/convert_pickle2npy.py \
        $cmumosi_pickle \
        $output_dir/feat/video/mmdatasdk_noalign \
        $output_dir/feat/audio/mmdatasdk_noalign \
        $output_dir/feat/text/mmdatasdk_noalign 

    # MMSA (https://github.com/thuiar/MMSA): FACET 35dim, COVEREP 74dim, BERT-Base 768dim
    python scripts/preproc/convert_pickle2npy.py \
        $cmumosi_pickle_mmsa \
        $output_dir/feat/video/mmsa_noalign \
        $output_dir/feat/audio/mmsa_noalign \
        $output_dir/feat/text/mmsa_noalign 
fi
