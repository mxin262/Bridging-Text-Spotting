# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from DiG.modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from DiG.models.transformer_layer import (
  get_pad_mask, get_subsequent_mask
)

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.,
                 use_learnable_pos_emb=False, use_mean_pooling=False, init_scale=0.001, return_feat_map=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask=None, bridge=None, rec_feat=None):
        # input preprocessing
        x = self.patch_embed(x)
        if bridge is not None:
            bridge_feat = bridge(rec_feat, mask, x)
        ## replace masked patches with mask_token
        B, N, C = x.shape
        if mask is not None:
            vis_mask = ~mask
            x = x * vis_mask.unsqueeze(-1) + self.mask_token.expand(B, N, -1) * mask.unsqueeze(-1)
        ## add position embedding
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        # encoder
        for blk in self.blocks:
            x = blk(x)
        if bridge is not None:
            x = self.norm(x+bridge_feat)
        else:
            x = self.norm(x)
        return x

    def forward(self, x, mask=None, bridge=None, rec_feat=None):
        x = self.forward_features(x, mask, bridge, rec_feat)
        x = self.head(x)
        return x


@register_model
def simmim_vit_tiny_patch4_32x128(pretrained=False, **kwargs):
  model = PretrainVisionTransformerEncoder(
      img_size=(32, 128), patch_size=4, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
      norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
  model.default_cfg = _cfg()
  return model

@register_model
def simmim_vit_small_patch4_32x128(pretrained=False, **kwargs):
  model = PretrainVisionTransformerEncoder(
      img_size=(32, 128), patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
      norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
  model.default_cfg = _cfg()
  return model

@register_model
def simmim_vit_base_patch4_32x128(pretrained=False, **kwargs):
  model = PretrainVisionTransformerEncoder(
      img_size=(32, 128), patch_size=4, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
      norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
  model.default_cfg = _cfg()
  return model