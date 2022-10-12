""" Transducer EEND model """
import torch
import torch.nn as nn
import yaml
import pdb

from mytorch.model.selective_decoder import SelectiveDecoder
from mytorch.model.unimodal_encoder import UniModalEncoder


class MultiModalNet(nn.Module):
    def __init__(self,
                 model_param,
                 ):
        super(MultiModalNet, self).__init__()
        self.params = model_param
        self.embed_size = model_param['video_encoder']['embed_size']

        input_dim_v, input_dim_a, input_dim_t = model_param['input_dims']
        # create encoders
        self.enc_v, self.enc_a, self.enc_t = None, None, None
        if input_dim_v:
            self.enc_v = UniModalEncoder(
                    input_dim_v,
                    self.params['video_encoder']['enc_size'],
                    self.params['video_encoder']['enc_layer'],
                    self.params['video_encoder']['dec_size'],
                    self.params['video_encoder']['dec_layer'],
                    self.params['video_encoder']['embed_size'],
                    self.params['video_encoder']['head_size'],
                    self.params['video_encoder']['dropout_rate'],
                    self.params['video_encoder']['feat_dropout_rate'],
                    self.params['video_encoder']['pooling']
                    )
        if input_dim_a:
            self.enc_a = UniModalEncoder(
                    input_dim_a,
                    self.params['audio_encoder']['enc_size'],
                    self.params['audio_encoder']['enc_layer'],
                    self.params['audio_encoder']['dec_size'],
                    self.params['audio_encoder']['dec_layer'],
                    self.params['audio_encoder']['embed_size'],
                    self.params['audio_encoder']['head_size'],
                    self.params['audio_encoder']['dropout_rate'],
                    self.params['audio_encoder']['feat_dropout_rate'],
                    self.params['audio_encoder']['pooling']
                    )
        if input_dim_t:
            self.enc_t = UniModalEncoder(
                    input_dim_t,
                    self.params['text_encoder']['enc_size'],
                    self.params['text_encoder']['enc_layer'],
                    self.params['text_encoder']['dec_size'],
                    self.params['text_encoder']['dec_layer'],
                    self.params['text_encoder']['embed_size'],
                    self.params['text_encoder']['head_size'],
                    self.params['text_encoder']['dropout_rate'],
                    self.params['text_encoder']['feat_dropout_rate'],
                    self.params['text_encoder']['pooling']
                    )
        # create decoder
        self.dec = SelectiveDecoder(
                self.params['video_encoder']['embed_size'],
                self.params['multimodal_decoder']['dec_size'],
                self.params['multimodal_decoder']['dec_layer'],
                self.params['output_dim'],
                self.params['multimodal_decoder']['head_size'],
                self.params['multimodal_decoder']['dropout_rate']
                )

    def forward(self, x, activation=None):
        """
        Args:
            x (torch.Tensor): input feature (B, T, in_dim)
            x_len (torch.Tensor): input feature length (B)

        Returns:
            loss (torch.Tensor): multi-label speaker activation loss (+ attractor loss)
        """
        h_v, att_v, h_a, att_a, h_t, att_t = None, None, None, None, None, None
        if self.enc_v:
            h_v, att_v = self.enc_v(x[0], x[1])
        if self.enc_a:
            h_a, att_a = self.enc_a(x[2], x[3])
        if self.enc_t:
            h_t, att_t = self.enc_t(x[4], x[5])
        
        y, att_dec = self.dec(h_v, h_a, h_t)
        
        if activation:
            y = activation(y)
        
        return y, (att_v, att_a, att_t, att_dec)

    def estimate(self, x):
        return self.forward(x, activation=torch.sigmoid)
