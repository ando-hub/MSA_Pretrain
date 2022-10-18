#!/bin/bash

stage=6
stop_stage=6

gpuid=3                # Use ${gpuid}-th GPU in feature extraction if $gpuid >= 0
extfeat_batchsize=4     # proc every $batchsize files in audio/text feature extraction

# cmumosei dataset path
#cmumosei_root='./data/src/CMU-MOSEI/Raw'
#cmumosei_label_csd='./data/src/CMU-MOSEI/CMU-MultimodalSDK/labels/CMU_MOSEI_Labels.csd'
#cmumosei_feat_mmdatasdk='./data/src/CMU-MOSEI/processed_data/cmu-mosei/seq_length_50/mosei_senti_data_noalign.pkl'
#cmumosei_feat_mmsa='./data/src/CMU-MOSEI/MMSA/Processed/unaligned_50.pkl'
cmumosei_root='/nfs/data/open/CMU-MOSEI/Raw'
cmumosei_label_csd='/nfs/data/open/CMU-MOSEI/CMU-MultimodalSDK/labels/CMU_MOSEI_Labels.csd'
cmumosei_feat_mmdatasdk='/nfs/data/open/CMU-MOSEI/processed_data/cmu-mosei/seq_length_50/mosei_senti_data_noalign.pkl'
cmumosei_feat_mmsa='/home/ando/work/220613_MMER_MMSA/data/MOSEI/Processed/unaligned_50.pkl'

output_dir='./data/dataset/cmumosei'
label_format='sentiment_regress'    # sentiment_regress, sentiment_class, or emo_class

# pre-trained encoder setup
video_encoder='CLIP'
audio_encoder='wavlm'
text_encoder='bert-large'
wavlm_model_path='./conf/pretrained_enc/WavLM-Large.pt'


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    # stage 0: download datasets
    # CMU-MOSEI raw dataset
    if [ ! -d $cmumosei_root ]; then
        echo "Download CMU-MOSEI raw dataset ..."
        mkdir -p $cmumosei_root
        wget -P $cmumosei_root/../ http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip
        unzip $cmumosei_root/../CMU_MOSEI.zip -d $cmumosei_root/../
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
        if [ ! -d ${cmumosei_feat_mmsa%/*} ]; then
            mkdir -p ${cmumosei_feat_mmsa%/*}
        fi
        echo "No exist CMU-MOSEI feat (MMSA) !"
        echo "Please access 'https://github.com/thuiar/MMSA', download 'MOSEI/Processed/unaligned_50.pkl' in 2.Datasets, and save it to ${cmumosei_feat_mmsa%/*}"
        exit 1
    fi
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    # stage 1: get utterance-level video/audio/text files
    echo "get interval info ..."
    python scripts/preproc/mosei_get_interval_list.py \
        $cmumosei_root/Transcript/Segmented/Combined \
        $cmumosei_label_csd \
        $output_dir/interval/cmumosei_interval_all.txt
    
    echo "get utterance-level video/audio files ..."
    python scripts/preproc/mosei_segment_video.py \
        $cmumosei_root/Videos/Full/Combined \
        $output_dir/interval/cmumosei_interval_all.txt \
        $output_dir/video/original \
        $output_dir/audio/original 

    echo "get utterance-level text files ..."
    python scripts/preproc/mosei_segment_text.py \
        $cmumosei_root/Transcript/Segmented/Combined \
        $output_dir/interval/cmumosei_interval_all.txt \
        $output_dir/text/original
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    # stage 2: create utterance-level labels
    echo "create utterance-level labels ..."
    python scripts/preproc/mosei_create_label.py \
        --label-format $label_format \
        $cmumosei_label_csd \
        $output_dir/interval/cmumosei_interval_all.txt \
        $output_dir/label/$label_format
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    # stage 3: feature extraction from video
    echo "frame resampling, face detection and feature extraction from video ..."
    python scripts/extfeat/extract_video_embedding.py \
        $output_dir/video/original \
        $output_dir/feat/video/pretrained_dbg2 \
        --gpuid $gpuid \
        --encoder-type $video_encoder \
        --fps 3 \
        --face-outd $output_dir/video/face_dbg2 \
        --image-outd $output_dir/video/image_dbg2 \
        --face-size 256 \
        --get-layer-results
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    # stage 4: feature extraction from audio
    echo "feature extraction from audio ..."
    python scripts/extfeat/extract_audio_embedding.py \
        $output_dir/audio/original \
        $output_dir/feat/audio/pretrained \
        --gpuid $gpuid \
        --encoder-type $audio_encoder \
        --wavlm-model-path $wavlm_model_path \
        -r 10 \
        --get-layer-results \
        --batchsize $extfeat_batchsize
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    # stage 5: feature extraction from text
    echo "tokenize, indexing and feature extraction from text ..."
    python scripts/extfeat/extract_text_embedding.py \
        $output_dir/text/original \
        $output_dir/feat/text/pretrained \
        --gpuid $gpuid \
        --encoder-type $text_encoder \
        --token-outd $output_dir/text/token_${text_encoder} \
        --get-layer-results \
        --batchsize $extfeat_batchsize
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    # 6. get conventional features for comparison
    # mmdatasdk (CMU-MOSEI official features): FACET 35dim, COVEREP 74dim, GloVe 300dim
    echo "mmdatasdk feature preparation ..."
    python scripts/preproc/mosei_convert_mmdatasdk_dataset.py \
        $cmumosei_feat_mmdatasdk \
        $output_dir/interval/cmumosei_interval_all.txt \
        $output_dir/feat/video/mmdatasdk_noalign \
        $output_dir/feat/audio/mmdatasdk_noalign \
        $output_dir/feat/text/mmdatasdk_noalign 

    # MMSA (https://github.com/thuiar/MMSA): FACET 35dim, COVEREP 74dim, BERT-Base 768dim
    echo "MMSA feature preparation ..."
    python scripts/preproc/convert_pickle2npy.py \
        $cmumosei_feat_mmsa \
        $output_dir/feat/video/mmsa_noalign \
        $output_dir/feat/audio/mmsa_noalign \
        $output_dir/feat/text/mmsa_noalign 
fi
