# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import random
import math
import numpy as np

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio, aug_ratio=0., num_view=1):
        self.num_view = num_view

        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)
        self.num_aug = int(self.num_mask * aug_ratio)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)

        if self.num_view > 1:
            masks = [mask]
            for i in range(self.num_view - 1):
                mask = np.hstack([
                    np.zeros(self.num_patches - self.num_mask),
                    np.ones(self.num_mask),
                ])
                np.random.shuffle(mask)
                masks.append(mask)
            mask = np.stack(masks) # [num_view, num_patches]
        return mask