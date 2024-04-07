from timm.models.registry import register_model
from timm.models import create_model

import DiG.modeling_finetune
import DiG.modeling_pretrain_vit

def create_encoder(args):

  encoder = create_model(
      "simmim_vit_small_patch4_32x128",
      pretrained=False,
      # num_classes=args.nb_classes,
      num_classes=0,
      drop_rate=0.0,
      drop_path_rate=0.1,
      attn_drop_rate=0.1,
      drop_block_rate=None,
      use_mean_pooling=False,
      init_scale=0.001,
      return_feat_map=not False,
  )
  
  return encoder