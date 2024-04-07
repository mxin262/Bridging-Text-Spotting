from audioop import bias
from builtins import NotImplementedError
from json import encoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial, reduce
from operator import mul

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from modeling_pretrain_vit import PretrainVisionTransformerEncoder
from modeling_finetune import _cfg, DropPath, Mlp

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, all_head_dim, bias=False)
        self.linear_k = nn.Linear(dim, all_head_dim, bias=False)
        self.linear_v = nn.Linear(dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim), requires_grad=False)
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, mask=None, return_attn_map=False):
        B, len_q, C = q.shape
        _, len_k, _ = k.size()

        q = F.linear(input=q, weight=self.linear_q.weight, bias=self.q_bias)
        k = F.linear(input=k, weight=self.linear_k.weight, bias=self.k_bias)
        v = F.linear(input=v, weight=self.linear_v.weight, bias=self.v_bias)

        q = q.reshape(B, len_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, len_k, self.num_heads, -1).permute(0, 2, 3, 1)
        v = v.reshape(B, len_k, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k

        if mask is not None:
            # [B, N] -> [B, num_heads, N, N]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        real_attn = attn.softmax(dim=-1)
        attn = self.attn_drop(real_attn)
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, len_q, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn_map:
            return x, real_attn
        else:
            return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, k=None, v=None, att_mask=None, return_attn_map=False):
        if k is None and v is None:
          x = self.norm1(x)
          k = x
          v = x
        else:
          x = self.norm1(x)
          k = self.norm1(k)
          v = self.norm1(v)

        if self.gamma_1 is None:
            if return_attn_map:
              attn_x, attn_map = self.attn(x, k, v, att_mask, return_attn_map)
            else:
              attn_x = self.attn(x, k, v, att_mask, return_attn_map)
            x = x + self.drop_path(attn_x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if return_attn_map:
              attn_x, attn_map = self.attn(x, k, v, att_mask, return_attn_map)
            else:
              attn_x = self.attn(x, k, v, att_mask, return_attn_map)
            x = x + self.drop_path(self.gamma_1 * attn_x)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        if return_attn_map:
          return x, attn_map
        else:
          return x

class PatchNet(nn.Module):
  def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
               num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
               drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.,
               use_learnable_pos_emb=False, train_with_ctx=False,
               num_windows=5, patch_shape=(8, 32), use_patch_transformer=False, hierarchical_num_windows=[1,2,4],):
    super().__init__()
    if use_patch_transformer:
      dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
      self.blocks = nn.ModuleList([
          Block(
              dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
              drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
              init_values=init_values)
          for i in range(depth)])
      self.norm =  norm_layer(embed_dim)
      self.apply(self._init_weights)

    self.num_windows = num_windows
    self.patch_shape = patch_shape
    self.use_patch_transformer = use_patch_transformer

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)

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
  
  def forward(self, seq_x, return_attn_map=False):
    B, _, C = seq_x.shape # [B, 8*32, C]

    x = seq_x.reshape(B, self.patch_shape[0], self.patch_shape[1], C).permute(0, 3, 1, 2) # [B, 8, 32, C]
    x = F.adaptive_avg_pool2d(x, (1, self.num_windows)).permute(0, 2, 3, 1).squeeze(1) # [B, num_windows, C]

    if self.use_patch_transformer:
      for blk in self.blocks:
        if return_attn_map:
          x, attn_map = blk(x, seq_x, seq_x, return_attn_map=True)
        else:
          x = blk(x, seq_x, seq_x)
      x = self.norm(x)
    if return_attn_map:
      return x, attn_map
    else:
      return x
      
class ConvPatchNet(nn.Module):
  def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
               num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
               drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.,
               use_learnable_pos_emb=False, train_with_ctx=False,
               num_windows=5, patch_shape=(8, 32), use_patch_transformer=False, hierarchical_num_windows=[1,2,4],):
    super().__init__()
    
    # The input size is (8, 32), we use 3 2-stride conv to extract features
    n_filter_list = [embed_dim, embed_dim*2, embed_dim*2, embed_dim*2] # (channels, 48, 96, 192, 384)  # hardcoding for now because that's what the paper used
    self.conv_layers = nn.Sequential(
      self.conv3x3_block(embed_dim, embed_dim),
      nn.MaxPool2d(kernel_size=2, stride=2), # [4, 16]
      self.conv3x3_block(embed_dim, int(embed_dim*1.5)),
      nn.MaxPool2d(kernel_size=2, stride=2), # [2, 8]
      self.conv3x3_block(int(embed_dim*1.5), embed_dim*2),
      nn.MaxPool2d(kernel_size=2, stride=2), # [1, 4]
      self.conv3x3_block(embed_dim*2, embed_dim*2),
      # nn.Conv2d(n_filter_list[3], embed_dim, stride=1, kernel_size=1, padding=0),
      # nn.BatchNorm2d(embed_dim), # [b, c, 1, 4]
    )
    self.patches2global = nn.Sequential(
      nn.Linear(embed_dim*2 * num_windows, embed_dim),
      nn.BatchNorm1d(embed_dim),
      nn.ReLU(inplace=True),
      nn.Linear(embed_dim, embed_dim),
      nn.BatchNorm1d(embed_dim, affine=False))

    self.num_windows = num_windows
    self.patch_shape = patch_shape
    self.use_patch_transformer = use_patch_transformer

  def conv3x3_block(self, in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

    block = nn.Sequential(
      conv_layer,
      nn.BatchNorm2d(out_planes),
      nn.ReLU(inplace=True),
    )
    return block
  
  def forward(self, seq_x, return_attn_map=False):
    B, _, C = seq_x.shape # [B, 8*32, C]

    x = seq_x.reshape(B, self.patch_shape[0], self.patch_shape[1], C).permute(0, 3, 1, 2) # [B, C, 8, 32]
    x = self.conv_layers(x)
    x = F.adaptive_avg_pool2d(x, (1, self.num_windows)).permute(0, 2, 3, 1).reshape(B, -1) # [B, num_windows, C]
    x = self.patches2global(x).unsqueeze(1)

    return x


class MoCo_ViT(nn.Module):
  def __init__(self,
               img_size=224, 
               patch_size=16, 
               in_chans=3, 
               encoder_num_classes=0, 
               encoder_embed_dim=768, 
               encoder_depth=12,
               encoder_num_heads=12,
               decoder_num_classes=768, 
               decoder_embed_dim=512, 
               decoder_depth=8,
               decoder_num_heads=8, 
               mlp_ratio=4., 
               qkv_bias=False, 
               qk_scale=None, 
               drop_rate=0., 
               attn_drop_rate=0.,
               drop_path_rate=0., 
               norm_layer=nn.LayerNorm, 
               init_values=0.,
               use_learnable_pos_emb=False,
               num_classes=0, # avoid the error from create_fn in timm
               mlp_dim=4096, # extra arguments for moco
               dim=256,
               T=1.0,
               num_windows=5,
               use_pixel_target=False,
               use_moco_target=True,
               encoder_type='vit', # 'vit' or 'resnet'
               queue_size=65536, #4096, # the size of memory banck
               patchnet_name='regular',
               label_smoothing=0.,
               use_pix_projector=True,): # add a small conv net before contrastive learning to give some localization information
    """
    dim: feature dimension (default: 256)
    mlp_dim: hidden dimension in MLPs (default: 4096)
    T: softmax temperature (default: 1.0)
    num_windows: the patch numbers of the feature maps
    use_patch_transformer: each patch is obtained via a transformer.
    use_image_slice: rather than slice image at feat-level, directly on the input image level.
    """
    super().__init__()

    self.T = T
    self.num_windows = num_windows
    self.use_pixel_target = use_pixel_target
    self.use_moco_target = use_moco_target
    self.label_smoothing = label_smoothing
    
    # encoder
    self.encoder = PretrainVisionTransformerEncoder(
              img_size=img_size, 
              patch_size=patch_size, 
              in_chans=in_chans, 
              num_classes=encoder_num_classes, 
              embed_dim=encoder_embed_dim, 
              depth=encoder_depth,
              num_heads=encoder_num_heads, 
              mlp_ratio=mlp_ratio, 
              qkv_bias=qkv_bias, 
              qk_scale=qk_scale, 
              drop_rate=drop_rate, 
              attn_drop_rate=attn_drop_rate,
              drop_path_rate=drop_path_rate, 
              norm_layer=norm_layer, 
              init_values=init_values,
              use_learnable_pos_emb=use_learnable_pos_emb,)

    # moco branch
    if use_moco_target:
      print('using moco branch.')
      self.momentum_encoder = PretrainVisionTransformerEncoder(
                img_size=img_size, 
                patch_size=patch_size, 
                in_chans=in_chans, 
                num_classes=encoder_num_classes, 
                embed_dim=encoder_embed_dim, 
                depth=encoder_depth,
                num_heads=encoder_num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale, 
                drop_rate=drop_rate, 
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate, 
                norm_layer=norm_layer, 
                init_values=init_values,
                use_learnable_pos_emb=use_learnable_pos_emb,)
        
      # xavier_uniform initialization
      # nn.init.xavier_uniform_(self.encoder.patch_embed.proj.weight)
      val = math.sqrt(6. / float(3 * reduce(mul, self.encoder.patch_embed.patch_size, 1) + encoder_embed_dim))
      nn.init.uniform_(self.encoder.patch_embed.proj.weight, -val, val)
      nn.init.zeros_(self.encoder.patch_embed.proj.bias)

      # freeze the patch embedding
      # self.encoder.patch_embed.proj.weight.requires_grad = False
      # self.encoder.patch_embed.proj.bias.requires_grad = False
      
      # remove the cls token and the last ln
      self.encoder.norm = nn.Identity()
      self.momentum_encoder.norm = nn.Identity()

      # projectors
      self.encoder_projection_layer = self._build_mlp(3, encoder_embed_dim, mlp_dim, dim)
      self.momentum_projection_layer = self._build_mlp(3, encoder_embed_dim, mlp_dim, dim)
      
      self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

      # get the patch features
      if patchnet_name == 'regular':
        patch_net = PatchNet
        use_patch_transformer = True
      elif patchnet_name == 'no_patchtrans':
        patch_net = PatchNet
        use_patch_transformer = False
      elif patchnet_name == 'conv':
        patch_net = ConvPatchNet
        use_patch_transformer = False
      else:
        raise NotImplementedError

      self.patch_extractor = patch_net(
        embed_dim=encoder_embed_dim,
        depth=2,
        num_heads=encoder_num_heads,
        num_windows=num_windows,
        patch_shape=self.encoder.patch_embed.patch_shape,
        use_patch_transformer=use_patch_transformer,)
      self.momentum_patch_extractor = patch_net(
        embed_dim=encoder_embed_dim,
        depth=2,
        num_heads=encoder_num_heads,
        num_windows=num_windows,
        patch_shape=self.encoder.patch_embed.patch_shape,
        use_patch_transformer=use_patch_transformer,)

      for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
        param_m.data.copy_(param_b.data)  # initialize
        param_m.requires_grad = False  # not update by gradient

      for param_b, param_m in zip(self.encoder_projection_layer.parameters(), self.momentum_projection_layer.parameters()):
        param_m.data.copy_(param_b.data)  # initialize
        param_m.requires_grad = False  # not update by gradient

      for param_b, param_m in zip(self.patch_extractor.parameters(), self.momentum_patch_extractor.parameters()):
        param_m.data.copy_(param_b.data)  # initialize
        param_m.requires_grad = False  # not update by gradient

    # mim branch
    if use_pixel_target:
      print('using mim branch.')
      if use_moco_target and use_pix_projector:
        self.pix_projector = self._build_mlp(3, encoder_embed_dim, 512, encoder_embed_dim)
        self.pix_projector_m = self._build_mlp(3, encoder_embed_dim, 512, encoder_embed_dim)
        
        for param_b, param_m in zip(self.pix_projector.parameters(), self.pix_projector_m.parameters()):
          param_m.data.copy_(param_b.data)  # initialize
          param_m.requires_grad = False  # not update by gradient
        
      self.pix_decoder = nn.Sequential(nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False),
                                        nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=False),
                                        nn.LayerNorm(decoder_embed_dim, eps=1e-6),
                                        nn.GELU(),
                                        nn.Linear(decoder_embed_dim, decoder_num_classes))
      
  @torch.no_grad()
  def _update_momentum_encoder(self, m):
      """Momentum update of the momentum encoder"""
      for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
          param_m.data = param_m.data * m + param_b.data * (1. - m)
      
      for param_b, param_m in zip(self.encoder_projection_layer.parameters(), self.momentum_projection_layer.parameters()):
          param_m.data = param_m.data * m + param_b.data * (1. - m)

      for param_b, param_m in zip(self.patch_extractor.parameters(), self.momentum_patch_extractor.parameters()):
          param_m.data = param_m.data * m + param_b.data * (1. - m)
          
      if hasattr(self, 'pix_projector'):
        for param_b, param_m in zip(self.pix_projector.parameters(), self.pix_projector_m.parameters()):
          param_m.data = param_m.data * m + param_b.data * (1. - m)
    
  def contrastive_loss(self, q, k, return_acc=False, temp=1., hard_k=None):
      # normalize
      q = nn.functional.normalize(q, dim=1)
      k = nn.functional.normalize(k, dim=1)
      # gather all targets
      k = concat_all_gather(k)
      # Einstein sum is more intuitive
      logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
      N = logits.shape[0]  # batch size per GPU
      labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
      # Return loss and accuracy
      if return_acc:
        accs = accuracy(logits, labels, topk=(1, 5))
        # return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T), accs
        return label_smooth_loss(logits.shape[-1], self.label_smoothing)(logits, labels) * (2 * self.T), accs
      else:
        # return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
        return label_smooth_loss(logits.shape[-1], self.label_smoothing)(logits, labels) * (2 * self.T)

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

  @torch.jit.ignore
  def no_weight_decay(self):
    return {'pos_embed', 'cls_token'}

  def forward(self, image, aug_image, vis_mask_pos, m, only_mim_on_ori_img=True,):
    out_dict = {}

    all_images = torch.cat([image, aug_image], dim=0)
    
    if not self.use_pixel_target:
      vis_mask_pos = None
    else:
      num_view = vis_mask_pos.size(1)
      vis_mask_pos = vis_mask_pos.permute(1, 0, 2).reshape(-1, vis_mask_pos.size(-1)) # [num_view * B, num_patches]
    
    # compute features
    if hasattr(self, 'pix_projector'):
      # add projector to masked image
      temp_encoder_output = self.encoder(all_images, vis_mask_pos)
      masked_enc_o, aug_enc_o = temp_encoder_output.chunk(2, dim=0)
      b, l, c = masked_enc_o.shape
      masked_enc_o = self.pix_projector(masked_enc_o.reshape(b*l, c))
      masked_enc_o = masked_enc_o.reshape(b, l, c)
      encoder_output = torch.cat([masked_enc_o, aug_enc_o], dim=0)
    else:
      encoder_output = self.encoder(all_images, vis_mask_pos)
      temp_encoder_output = encoder_output.clone()
    
    if self.use_moco_target:
      patches = self.patch_extractor(encoder_output)

      b, l, c = patches.shape
      patches = patches.reshape(b*l, c)
      qs = self.encoder_projection_layer(patches)
      qs = self.predictor(qs)
      qs = qs.reshape(b, l, -1)

      q1, q2 = qs.chunk(2, dim=0)
      q1 = q1.view(-1, q1.size(-1))
      q2 = q2.view(-1, q2.size(-1))

      with torch.no_grad(): # no gradient
        self._update_momentum_encoder(m) # update the momentum encoder

        # compute momentum features as targets
        # add projector to masked image
        if hasattr(self, 'pix_projector'):
          temp_momentum_encoder_output = self.momentum_encoder(all_images, vis_mask_pos)
          masked_enc_o_m, aug_enc_o_m = temp_momentum_encoder_output.chunk(2, dim=0)
          b, l, c = masked_enc_o_m.shape
          masked_enc_o_m = self.pix_projector_m(masked_enc_o_m.reshape(b*l, c))
          masked_enc_o_m = masked_enc_o_m.reshape(b, l, c)
          momentum_encoder_output = torch.cat([masked_enc_o_m, aug_enc_o_m], dim=0)
        else:
          momentum_encoder_output = self.momentum_encoder(all_images, vis_mask_pos)

        momentum_patches = self.momentum_patch_extractor(momentum_encoder_output)
        
        b, l, c = momentum_patches.shape
        momentum_patches = momentum_patches.reshape(b*l, c)
        ks = self.momentum_projection_layer(momentum_patches)
        ks = ks.reshape(b, l, -1)

        k1, k2 = ks.chunk(2, dim=0)
        k1 = k1.view(-1, k1.size(-1))
        k2 = k2.view(-1, k2.size(-1))
      
      contra_loss1, (q1_acc1, q1_acc5) = self.contrastive_loss(q1, k2, return_acc=True, temp=self.T, hard_k=None)
      contra_loss2, (q2_acc1, q2_acc5) = self.contrastive_loss(q2, k1, return_acc=True, temp=self.T, hard_k=None)
      out_dict['contra_loss'] = contra_loss1 + contra_loss2
      # Accuracy
      out_dict['q1_acc1'] = q1_acc1
      out_dict['q1_acc5'] = q1_acc5
      out_dict['q2_acc1'] = q2_acc1
      out_dict['q2_acc5'] = q2_acc5

    if self.use_pixel_target:
      decoder_output = self.pix_decoder(temp_encoder_output)

      B, _, C = decoder_output.shape
      
      dec_out_list = list(decoder_output.chunk(num_view, dim=0))
      vis_mask_pos_list = list(vis_mask_pos.chunk(num_view, dim=0))

      if only_mim_on_ori_img:
        vis_out = dec_out_list[0][vis_mask_pos_list[0]].reshape(B//2, -1, C)
        out_dict['vis_out'] = [vis_out]
      else:
        vis_out_list = []
        for dec_out_, vis_mask_pos_ in zip(dec_out_list, vis_mask_pos_list):
          vis_out_list.append(dec_out_[vis_mask_pos_].reshape(B//2, -1, C))
        out_dict['vis_out'] = vis_out_list
                  
    return out_dict

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class label_smooth_loss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, focal_factor=0.):
        super(label_smooth_loss, self).__init__()
        eps = smoothing / num_classes
        self.negative = eps
        self.positive = (1 - smoothing) + eps
        self.focal_factor = focal_factor
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.negative)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.positive)
        
        # loss = torch.sum(-true_dist * pred, dim=1) * ((1 - torch.exp(torch.sum(true_dist * pred, dim=1))) ** self.focal_factor)
        # return loss.mean()
        return torch.sum(-true_dist * pred, dim=1).mean()

# small
@register_model
def pretrain_moco_ori_vit_small_patch4_32x128(pretrained=False, **kwargs):
    model = MoCo_ViT(
        img_size=(32, 128),
        patch_size=4,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=48,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_pixel_target=False,
        use_moco_target=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_simmim_ori_vit_small_patch4_32x128(pretrained=False, **kwargs):
    model = MoCo_ViT(
        img_size=(32, 128),
        patch_size=4,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=48,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_pixel_target=True,
        use_moco_target=False,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_simmim_moco_ori_vit_small_patch4_32x128(pretrained=False, **kwargs):
    model = MoCo_ViT(
        img_size=(32, 128),
        patch_size=4,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=48,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_pixel_target=True,
        use_moco_target=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

# tiny
@register_model
def pretrain_moco_ori_vit_tiny_patch4_32x128(pretrained=False, **kwargs):
    model = MoCo_ViT(
        img_size=(32, 128),
        patch_size=4,
        encoder_embed_dim=192,
        encoder_depth=12,
        encoder_num_heads=3,
        encoder_num_classes=0,
        decoder_num_classes=48,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_pixel_target=False,
        use_moco_target=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_simmim_ori_vit_tiny_patch4_32x128(pretrained=False, **kwargs):
    model = MoCo_ViT(
        img_size=(32, 128),
        patch_size=4,
        encoder_embed_dim=192,
        encoder_depth=12,
        encoder_num_heads=3,
        encoder_num_classes=0,
        decoder_num_classes=48,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_pixel_target=True,
        use_moco_target=False,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_simmim_moco_ori_vit_tiny_patch4_32x128(pretrained=False, **kwargs):
    model = MoCo_ViT(
        img_size=(32, 128),
        patch_size=4,
        encoder_embed_dim=192,
        encoder_depth=12,
        encoder_num_heads=3,
        encoder_num_classes=0,
        decoder_num_classes=48,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_pixel_target=True,
        use_moco_target=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

# base
@register_model
def pretrain_simmim_moco_ori_vit_base_patch4_32x128(pretrained=False, **kwargs):
    model = MoCo_ViT(
        img_size=(32, 128),
        patch_size=4,
        encoder_embed_dim=512,
        encoder_depth=12,
        encoder_num_heads=8,
        encoder_num_classes=0,
        decoder_num_classes=48,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_pixel_target=True,
        use_moco_target=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_simmim_ori_vit_base_patch4_32x128(pretrained=False, **kwargs):
    model = MoCo_ViT(
        img_size=(32, 128),
        patch_size=4,
        encoder_embed_dim=512,
        encoder_depth=12,
        encoder_num_heads=8,
        encoder_num_classes=0,
        decoder_num_classes=48,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_pixel_target=True,
        use_moco_target=False,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_moco_ori_vit_base_patch4_32x128(pretrained=False, **kwargs):
    model = MoCo_ViT(
        img_size=(32, 128),
        patch_size=4,
        encoder_embed_dim=512,
        encoder_depth=12,
        encoder_num_heads=8,
        encoder_num_classes=0,
        decoder_num_classes=48,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_pixel_target=False,
        use_moco_target=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
