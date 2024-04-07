import torch.nn as nn

from .encoder import *
from .decoder import *
from .attn_decoder import AttentionRecognitionHead
from DiG.models import encoder

class CTCRecModel(nn.Module):
  def __init__(self, args):
    super(CTCRecModel, self).__init__()

    self.encoder = create_encoder(args)
    d_embedding = 512
    self.ctc_classifier = nn.Sequential(nn.Linear(self.encoder.num_features, d_embedding),
                                        nn.LayerNorm(d_embedding, eps=1e-6),
                                        nn.GELU(),
                                        nn.Linear(d_embedding, args.nb_classes + 1))

    # some function and variable should be inherited.
    self.patch_embed = self.encoder.patch_embed
    self.pos_embed = self.encoder.pos_embed

  def no_weight_decay(self):
    skip_weight_decay_list = self.encoder.no_weight_decay()
    return {'encoder.' + item for item in skip_weight_decay_list}

  def get_num_layers(self):
    return self.encoder.get_num_layers()

  def forward(self, x):
    x, tgt, tgt_lens = x
    enc_x = self.encoder(x)

    B, N, C = enc_x.shape
    reshaped_enc_x = enc_x.view(B, *self.encoder.patch_embed.patch_shape, C).mean(1)
    ctc_logit = self.ctc_classifier(reshaped_enc_x)

    return ctc_logit

class AttnRecModel(nn.Module):
  def __init__(self, args):
    super(AttnRecModel, self).__init__()

    self.encoder = create_encoder(args)
    self.decoder = AttentionRecognitionHead(
                      num_classes=args.nb_classes,
                      in_planes=self.encoder.num_features,
                      sDim=512,
                      attDim=512,
                      max_len_labels=args.max_len) 

    # some function and variable should be inherited.
    self.patch_embed = self.encoder.patch_embed
    self.pos_embed = self.encoder.pos_embed

    # 1d or 2d features
    self.use_1d_attdec = args.use_1d_attdec
    self.beam_width = getattr(args, 'beam_width', 0)

  def no_weight_decay(self):
    skip_weight_decay_list = self.encoder.no_weight_decay()
    return {'encoder.' + item for item in skip_weight_decay_list}

  def get_num_layers(self):
    return self.encoder.get_num_layers()

  def forward(self, x):
    x, tgt, tgt_lens = x
    enc_x = self.encoder(x)

    dec_output, _ = self.decoder((enc_x, tgt, tgt_lens))
    return dec_output, None, None, None

class RecModel(nn.Module):
  def __init__(self, args):
    super(RecModel, self).__init__()

    self.encoder = create_encoder(args)
    self.decoder = create_decoder(args)
    # if args.decoder_name == 'small_tf_decoder':
    #   d_embedding = 384
    # else:  
    #   d_embedding = 512
    d_embedding = self.decoder.d_embedding
    self.linear_norm = nn.Sequential(
      nn.Linear(self.encoder.num_features, d_embedding),
      nn.LayerNorm(d_embedding),
    )

    # some function and variable should be inherited.
    self.patch_embed = self.encoder.patch_embed
    self.pos_embed = self.encoder.pos_embed

    # target embedding is used in both encoder and decoder
    if hasattr(self.encoder, 'insert_sem'):
      if self.encoder.insert_sem:
        self.trg_word_emb = nn.Embedding(
          args.nb_classes + 1, d_embedding
        )
        self.insert_sem = True
      else:
        self.trg_word_emb = None
        self.insert_sem = False
    else:
      self.trg_word_emb = None
      self.insert_sem = False

    # 1d or 2d features
    self.use_1d_attdec = False
    self.beam_width = getattr(args, 'beam_width', 0)
    
    # add feat projector
    self.use_feat_distill = getattr(args, 'use_feat_distill', False)
    if self.use_feat_distill:
      self.feat_proj = self._build_mlp(3, self.encoder.num_features, 4096, self.encoder.num_features)

  def no_weight_decay(self):
    skip_weight_decay_list = self.encoder.no_weight_decay()
    return {'encoder.' + item for item in skip_weight_decay_list}

  def get_num_layers(self):
    return self.encoder.get_num_layers()

  def forward(self, x, bridge):
    x, tgt, tgt_lens, rec_feat = x
    if self.insert_sem:
      enc_x, rec_score = self.encoder((x, self.trg_word_emb), bridge=bridge, rec_feat=rec_feat)
    else:
      enc_x = self.encoder(x, bridge=bridge, rec_feat=rec_feat)

    # maybe a multi-label branch is added.
    if isinstance(enc_x, tuple):
      cls_logit, enc_x, cls_logit_attn_maps = enc_x
    else:
      cls_logit = None
      cls_logit_attn_maps = None

    if not self.training:
      tgt = None
      tgt_lens = None
    # only use multi-label loss
    if enc_x is None and cls_logit is not None:
      # no decoder
      return None, cls_logit, None, None
    
    # 1d or 2d features for decoder
    if self.use_1d_attdec:
      B, N, C = enc_x.shape
      enc_x = enc_x.view(B, *self.encoder.patch_embed.patch_shape, C).mean(1)

    dec_in = self.linear_norm(enc_x)
    dec_output, dec_attn_maps = self.decoder(dec_in,
                                             dec_in,
                                             targets=tgt,
                                             tgt_lens=tgt_lens,
                                             train_mode=self.training,
                                             cls_query_attn_maps=cls_logit_attn_maps,
                                             trg_word_emb=self.trg_word_emb,
                                             beam_width=self.beam_width,)
    
    # feat distillation
    if self.use_feat_distill and self.training:
      b, l, c = enc_x.shape
      s_feat = self.feat_proj(enc_x.reshape(b*l, c))
      s_feat = s_feat.reshape(b, l, c)
      # s_feat = self.feat_proj(enc_x)
      return dec_output, s_feat
    
    # return dec_output, None, None, None
    return dec_output, None, None, dec_attn_maps
    
    B, len_q, len_k = dec_attn_maps.shape
    # dec_attn_maps = dec_attn_maps.view(B, len_q, *self.patch_embed.patch_shape)
    # dec_attn_maps = dec_attn_maps[:, :, :self.patch_embed.num_patches].view(B, len_q, *self.patch_embed.patch_shape)
    dec_attn_maps = None
    if self.insert_sem:
      return dec_output, rec_score, dec_attn_maps
    if cls_logit is not None:
      # reshpe attn_maps to spatial version
      return dec_output, cls_logit, cls_logit_attn_maps, dec_attn_maps
    else:
      return dec_output, None, None, dec_attn_maps

  def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True, use_conv=False):
      mlp = []
      for l in range(num_layers):
          dim1 = input_dim if l == 0 else mlp_dim
          dim2 = output_dim if l == num_layers - 1 else mlp_dim

          if use_conv:
            mlp.append(nn.Conv1d(dim1, dim2, 1, bias=False))
          else:
            mlp.append(nn.Linear(dim1, dim2, bias=False))

          if l < num_layers - 1:
              mlp.append(nn.BatchNorm1d(dim2))
              mlp.append(nn.ReLU(inplace=True))
          elif last_bn:
              # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
              # for simplicity, we further removed gamma in BN
              mlp.append(nn.BatchNorm1d(dim2, affine=False))

      return nn.Sequential(*mlp)

class MimRecModel(nn.Module):
  def __init__(self, args):
    super(MimRecModel, self).__init__()

    self.encoder = create_encoder(args)
    self.rec_decoder = create_decoder(args)
    self.pix_encoder_to_decoder = nn.Linear(self.encoder.num_features, 192, bias=False)
    self.decoder = nn.Sequential(nn.Linear(192, 192, bias=False),
                                     nn.LayerNorm(192, eps=1e-6),
                                     nn.GELU(),
                                     nn.Linear(192, 48))
    self.use_mim_loss = args.mim_sample_ratio > 0.
    self.use_mim_proj = args.use_mim_proj
    if self.use_mim_proj:
      self.mim_proj = nn.Sequential(nn.Linear(self.encoder.num_features, self.encoder.num_features * 2),
                                nn.LayerNorm(self.encoder.num_features * 2, eps=1e-6),
                                nn.GELU(),
                                nn.Linear(self.encoder.num_features * 2, self.encoder.num_features),
                                nn.LayerNorm(self.encoder.num_features, eps=1e-6),)
      # self.mim_proj = nn.Sequential(nn.Linear(self.encoder.num_features, self.encoder.num_features),
      #                           nn.LayerNorm(self.encoder.num_features, eps=1e-6),
      #                           nn.GELU(),
      #                           nn.Linear(self.encoder.num_features, self.encoder.num_features),
      #                           nn.LayerNorm(self.encoder.num_features, eps=1e-6),)
    # if args.decoder_name == 'small_tf_decoder':
    #   d_embedding = 384
    # else:  
    #   d_embedding = 512
    d_embedding = self.rec_decoder.d_embedding
    self.linear_norm = nn.Sequential(
      nn.Linear(self.encoder.num_features, d_embedding),
      nn.LayerNorm(d_embedding),
    )

    # some function and variable should be inherited.
    self.patch_embed = self.encoder.patch_embed
    self.pos_embed = self.encoder.pos_embed

  def no_weight_decay(self):
    skip_weight_decay_list = self.encoder.no_weight_decay()
    return {'encoder.' + item for item in skip_weight_decay_list}

  def get_num_layers(self):
    return self.encoder.get_num_layers()

  def forward(self, x):
    out_dict = {}

    if len(x) == 5:
      x, mask, num_mim_samples, tgt, tgt_lens = x
    else:
      x, tgt, tgt_lens = x
      mask = None
      num_mim_samples = 0
    # enc_x = self.encoder(x, mask)

    # mim loss
    if self.use_mim_loss:
      if self.use_mim_proj:
        temp_enc_x = self.encoder(x, mask)
        mim_enc_x = temp_enc_x[:num_mim_samples]
        mim_enc_x = self.mim_proj(mim_enc_x)
        enc_x = torch.cat([mim_enc_x, temp_enc_x[num_mim_samples:]], dim=0)
      else:
        temp_enc_x = self.encoder(x, mask)
        enc_x = temp_enc_x.clone()
      pix_dec_input = self.pix_encoder_to_decoder(temp_enc_x)
      pix_dec_output = self.decoder(pix_dec_input)
      out_dict['pix_pred'] = pix_dec_output
    else:
      enc_x = self.encoder(x, mask)

    # recognition      
    if not self.training:
      tgt = None
      tgt_lens = None

    enc_x = self.linear_norm(enc_x)
    dec_output, dec_attn_maps = self.rec_decoder(enc_x,
                                             enc_x,
                                             targets=tgt,
                                             tgt_lens=tgt_lens,
                                             train_mode=self.training,
                                             cls_query_attn_maps=None,
                                             trg_word_emb=None,)
    out_dict['rec_pred'] = dec_output
    return out_dict
