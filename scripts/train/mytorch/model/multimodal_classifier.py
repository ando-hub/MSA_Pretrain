""" Transducer EEND model """
import torch
import torch.nn as nn
import yaml
import pdb

from mytorch.model.multimodal_decoder import MultiModalDecoder
from mytorch.model.unimodal_encoder import UniModalEncoder


class MultiModalClassifier(nn.Module):
    def __init__(self,
                 model_param,
                 ):
        super(MultiModalClassifier, self).__init__()
        self.params = model_param
        self.embed_size = model_param['video_encoder']['embed_size']

        input_dim_v, input_dim_a, input_dim_t = model_param['input_dims']
        if 'input_layers' in model_param:
            input_layer_v, input_layer_a, input_layer_t = model_param['input_layers']
        else:
            input_layer_v, input_layer_a, input_layer_t = 1, 1, 1
        # create encoders
        self.enc_v, self.enc_a, self.enc_t = None, None, None
        if input_dim_v:
            if 'input_norm' in self.params['video_encoder']:
                input_norm_v = self.params['video_encoder']['input_norm']
            else:
                input_norm_v = False
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
                    self.params['video_encoder']['pooling'],
                    input_norm_v,
                    input_layer_v
                    )
        if input_dim_a:
            if 'input_norm' in self.params['audio_encoder']:
                input_norm_a = self.params['audio_encoder']['input_norm']
            else:
                input_norm_a = False
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
                    self.params['audio_encoder']['pooling'],
                    input_norm_a,
                    input_layer_a
                    )
        if input_dim_t:
            if 'input_norm' in self.params['text_encoder']:
                input_norm_t = self.params['text_encoder']['input_norm']
            else:
                input_norm_t = False
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
                    self.params['text_encoder']['pooling'],
                    input_norm_t,
                    input_layer_t
                    )
        # create decoder
        self.dec = MultiModalDecoder(
                self.params['video_encoder']['embed_size'],
                self.params['multimodal_decoder']['dec_size'],
                self.params['multimodal_decoder']['dec_layer'],
                self.params['output_dim'],
                self.params['multimodal_decoder']['head_size'],
                self.params['multimodal_decoder']['dropout_rate'],
                self.params['multimodal_decoder']['pooling'],
                self.params['loss']['lossfunc'],
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

        nbat = [h.shape[0] for h in [h_v, h_a, h_t] if h is not None][0]
        device = [h.device for h in [h_v, h_a, h_t] if h is not None][0]
        h_v = h_v if h_v is not None else torch.zeros((nbat, self.embed_size)).float().to(device)
        h_a = h_a if h_a is not None else torch.zeros((nbat, self.embed_size)).float().to(device)
        h_t = h_t if h_t is not None else torch.zeros((nbat, self.embed_size)).float().to(device)

        h = torch.cat((h_v.unsqueeze(1), h_a.unsqueeze(1), h_t.unsqueeze(1)), 1)    # [B, 3, D]
        y, att_dec = self.dec(h)

        if activation:
            y = activation(y)

        return y, (att_v, att_a, att_t, att_dec), (h_v, h_a, h_t)

    def estimate(self, x):
        return self.forward(x, activation=torch.sigmoid)[0]

    def get_embeddings(self, x):
        h_v, att_v, h_a, att_a, h_t, att_t = None, None, None, None, None, None
        if self.enc_v:
            h_v, att_v = self.enc_v(x[0], x[1])
        if self.enc_a:
            h_a, att_a = self.enc_a(x[2], x[3])
        if self.enc_t:
            h_t, att_t = self.enc_t(x[4], x[5])
        return (h_v, h_a, h_t)

