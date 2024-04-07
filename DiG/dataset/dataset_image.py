# this is just a copy of `dataset_lmdb.py`
# only need to modify the label to None
from os import replace
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch

import lmdb
import string
import six
from PIL import Image, ImageFile
import cv2
import random

from transforms import CVColorJitter, CVDeterioration, CVGeometry

from imgaug import augmenters as iaa
ImageFile.LOAD_TRUNCATED_IMAGES = True
cv2.setNumThreads(0) # cv2's multiprocess will impact the dataloader's workers.

class AloneImageLmdb(Dataset):
  def __init__(self, root, voc_type, max_len, num_samples, transform, use_aug=True):
    super(AloneImageLmdb, self).__init__()

    env = lmdb.open(root, max_readers=32, readonly=True)
    with env.begin() as txn:
      self.nSamples = int(txn.get(b"num-samples"))
    self.nSamples = min(self.nSamples, num_samples)
    env.close()
    del txn, env

    self.root = root
    self.max_len = max_len
    self.num_samples = num_samples
    self.transform = transform
    self.use_aug = use_aug

    if use_aug:
      self.augmentor = self.sequential_aug()
      mean = std = 0.5
      self.aug_transformer = transforms.Compose([
            transforms.Resize((32, 128), interpolation=3),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

    # Generate vocabulary
    assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    self.classes = self._find_classes(voc_type)
    self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
    self.idx_to_class = dict(zip(range(len(self.classes)), self.classes))
    self.use_lowercase = (voc_type == 'LOWERCASE')

  def _find_classes(self, voc_type, EOS='EOS',
                    PADDING='PADDING', UNKNOWN='UNKNOWN'):
    '''
    voc_type: str: one of 'LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'
    '''
    voc = None
    types = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    if voc_type == 'LOWERCASE':
      # voc = list(string.digits + string.ascii_lowercase)
      voc = list('0123456789abcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    elif voc_type == 'ALLCASES':
      voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
      voc = list(string.printable[:-6])
    else:
      raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    return voc

  def __len__(self):
    return self.nSamples

  def sequential_aug(self):
    aug_transform = transforms.Compose([
      iaa.Sequential(
        [
          iaa.SomeOf((2, 5),
          [
            iaa.LinearContrast((0.5, 1.0)),
            iaa.GaussianBlur((0.5, 1.5)),
            iaa.Crop(percent=((0, 0.3),
                              (0, 0.0),
                              (0, 0.3),
                              (0, 0.0)),
                              keep_size=True),
            iaa.Crop(percent=((0, 0.0),
                              (0, 0.1),
                              (0, 0.0),
                              (0, 0.1)),
                              keep_size=True),
            iaa.Sharpen(alpha=(0.0, 0.5),
                        lightness=(0.0, 0.5)),
            # iaa.AdditiveGaussianNoise(scale=(0, 0.15*255), per_channel=True),
            iaa.Rotate((-10, 10)),
            # iaa.Cutout(nb_iterations=1, size=(0.15, 0.25), squared=True),
            iaa.PiecewiseAffine(scale=(0.03, 0.04), mode='edge'),
            iaa.PerspectiveTransform(scale=(0.05, 0.1)),
            iaa.Solarize(1, threshold=(32, 128), invert_above_threshold=0.5, per_channel=False),
            iaa.Grayscale(alpha=(0.0, 1.0)),
          ],
          random_order=True)
        ]
      ).augment_image,
    ])
    return aug_transform

  def open_lmdb(self):
    self.env = lmdb.open(self.root, readonly=True, create=False)
    self.txn = self.env.begin(buffers=True)

  def __getitem__(self, index):
    if not hasattr(self, 'txn'):
      self.open_lmdb()

    # Load image
    assert index <= len(self), 'index range error'
    index += 1
    img_key = b'image-%09d' % index
    imgbuf = self.txn.get(img_key)

    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    try:
      img = Image.open(buf).convert('RGB')
    except IOError:
      print('Corrupted image for %d' % index)
      return self[index + 1]
    
    if self.use_aug:
      # augmentation
      aug_img = self.augmentor(np.asarray(img))
      aug_img = Image.fromarray(np.uint8(aug_img))
      aug_img = self.aug_transformer(aug_img)

    assert self.transform is not None
    trans = self.transform(img)
    assert isinstance(trans, tuple)
    img, vis_mask = trans
    
    # To be compatible with the return of `dataset_lmdb.py`
    if self.use_aug:
      return (img, aug_img, vis_mask), np.ones(1), np.ones(1)
    else:
      return (img, vis_mask, np.ones(1)), np.ones(1), np.ones(1)