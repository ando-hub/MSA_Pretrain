#!/bin/bash

stage=0
stop_stage=100

gpuid=-1                # Use ${gpuid}-th GPU in feature extraction if $gpuid >= 0
extfeat_batchsize=4     # proc every $batchsize files in audio/text feature extraction

# cmumosei dataset path
cmumosei_root='/nfs/data/open/CMU-MOSEI/Raw'
cmumosei_label_csd='/nfs/data/open/CMU-MOSEI/CMU-MultimodalSDK/labels/CMU_MOSEI_Labels.csd'
cmumosei_feat_mmdatasdk='/nfs/data/open/CMU-MOSEI/processed_data/cmu-mosei/seq_length_50/mosei_senti_data_noalign.pkl'
cmumosei_feat_mmsa='/home/ando/work/220613_MMER_MMSA/data/MOSEI/Processed/unaligned_50.pkl'

output_dir='./data/dataset/cmumosei'
label_format='sentiment_regress'    # sentiment_regress, sentiment_class, or emo_class

# pre-trained encoder setup
video_encoder='CLIP'
audio_encoder='WavLM'
text_encoder='BERT_Large'


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    # stage 0: download datasets
    # CMU-MOSEI raw dataset
    if [ ! -d $cmumosei_root ]; then
        echo "Download CMU-MOSEI raw dataset ..."
        mkdir -p $cmumosei_root
        wget -P $cmumosei_root/../ http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip
        unzip $cmumosei_root/../CMU_MOSEI.zip -d $cmumosei_root
    fi
    
    # CMU-MOSEI labels
    if [ ! -f $cmumosei_label_csd ]; then
        echo "Download CMU-MOSEI label ..."
        if [ ! -d ${cmumosei_label_csd%/*} ]; then
            mkdir -p ${cmumosei_label_csd%/*}
        fi
        wget -P ${cmumosei_label_csd%/*} http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd 
    fi
    
    # CMU-MOSEI features (mmdatasdk)
    if [ ! -f $cmumosei_feat_mmdatasdk ]; then
        echo "Download CMU-MOSEI feat (mmdatasdk) ..."
        if [ ! -d ${cmumosei_feat_mmdatasdk%/*} ]; then
            mkdir -p ${cmumosei_feat_mmdatasdk%/*}
        fi
        wget -P ${cmumosei_feat_mmdatasdk%/*} http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosei/seq_length_50/mosei_senti_data_noalign.pkl
    fi

    # CMU-MOSEI features (mmsa)
    if [ ! -f $cmumosei_feat_mmsa ]; then
        echo "No exist CMU-MOSEI feat (MMSA) !"
        echo "Please access 'https://github.com/thuiar/MMSA', download 'MOSEI/Processed/unaligned_50.pkl' in 2.Datasets, and save it to ${cmumosei_feat_mmsa%/*}"
        exit 1
    fi
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    # stage 1: get utterance-level video/audio/text files

    # get interval info
    python scripts/preproc/mosei_get_interval_list.py \
        $cmumosei_root/Transcript/Segmented/Combined \
        $cmumosei_label_csd \
        $output_dir/interval/cmumosei_interval_all.txt
    
    # get utterance-level video/audio files from entire videos 
    python scripts/preproc/mosei_segment_video.py \
        $cmumosei_root/Videos/Full/Combined \
        $output_dir/interval/cmumosei_interval_all.txt \
        $output_dir/video/original \
        $output_dir/audio/original 

    # get utterance-level text files from entire transcriptions 
    python scripts/preproc/mosei_segment_text.py \
        $cmumosei_root/Transcript/Segmented/Combined \
        $output_dir/interval/cmumosei_interval_all.txt \
        $output_dir/text/original
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    # stage 2: create utterance-level labels
    python scripts/preproc/mosei_create_label.py \
        --label-format $label_format \
        $cmumosei_label_csd \
        $output_dir/interval/cmumosei_interval_all.txt \
        $output_dir/label/$label_format
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
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
        --face-size 256 \
        --get-layer-results
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    # 4. feature extraction from audio
    python scripts/extfeat/extract_audio_embedding.py \
        $output_dir/audio/original \
        $output_dir/feat/audio/tmp_$audio_encoder \
        --gpuid $gpuid \
        --model-type $audio_encoder \
        --get-layer-results \
        --batchsize $extfeat_batchsize
    
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
    # mmdatasdk (CMU-MOSEI official features): FACET 35dim, COVEREP 74dim, GloVe 300dim
    python scripts/preproc/mosei_convert_mmdatasdk_dataset.py \
        $cmumosei_pickle \
        $output_dir/interval/cmumosei_interval_all.txt \
        $output_dir/feat/video/mmdatasdk_noalign \
        $output_dir/feat/audio/mmdatasdk_noalign \
        $output_dir/feat/text/mmdatasdk_noalign 

    # MMSA (https://github.com/thuiar/MMSA): FACET 35dim, COVEREP 74dim, BERT-Base 768dim
    python scripts/preproc/convert_pickle2npy.py \
        $cmumosei_pickle_mmsa \
        $output_dir/feat/video/mmsa_noalign \
        $output_dir/feat/audio/mmsa_noalign \
        $output_dir/feat/text/mmsa_noalign 
fi
