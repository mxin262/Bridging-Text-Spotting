from torch.utils.data import Dataset
from torchvision import transforms
import torch

import lmdb
import string
import six
import numpy as np
from PIL import Image, ImageFile
import cv2

from transforms import CVColorJitter, CVDeterioration, CVGeometry

from imgaug import augmenters as iaa
ImageFile.LOAD_TRUNCATED_IMAGES = True
cv2.setNumThreads(0) # cv2's multiprocess will impact the dataloader's workers.

class ImageLmdb(Dataset):
  def __init__(self, root, voc_type, max_len, num_samples, transform,
               use_aug=False, use_abi_aug=False, use_color_aug=False):
    super(ImageLmdb, self).__init__()
    self.env = lmdb.open(root, max_readers=32, readonly=True)
    self.txn = self.env.begin()
    self.nSamples = int(self.txn.get(b"num-samples"))

    num_samples = num_samples if num_samples > 1 else int(self.nSamples * num_samples)
    self.nSamples = int(min(self.nSamples, num_samples))

    self.root = root
    self.max_len = max_len
    self.transform = transform
    self.use_aug = use_aug
    self.use_abi_aug = use_abi_aug
    self.use_color_aug = use_color_aug
    if use_aug:
      if use_abi_aug:
        mean = std = 0.5
        self.augment_abi = transforms.Compose([
              CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
              CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
              CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25),
              transforms.Resize((32, 128), interpolation=3),
              transforms.ToTensor(),
              transforms.Normalize(
                  mean=torch.tensor(mean),
                  std=torch.tensor(std))
          ])
      else:
        # augmentation following seqCLR
        if use_color_aug:
          self.augmentor = self.color_aug()
        else:
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
  
  def color_aug(self):
    aug_transform = transforms.Compose([
      iaa.Sequential(
        [
          iaa.SomeOf((2, 5),
          [
            iaa.LinearContrast((0.5, 1.0)),
            iaa.GaussianBlur((0.5, 1.5)),
            iaa.Sharpen(alpha=(0.0, 0.5),
                        lightness=(0.0, 0.5)),
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
    # self.txn = self.env.begin(buffers=True)
    self.txn = self.env.begin()

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
    
    # Load label
    label_key = b'label-%09d' % index
    word = self.txn.get(label_key).decode()
    if self.use_lowercase:
      word = word.lower()

    if len(word) + 1 >= self.max_len:
      # print('%s is too long.' % word)
      return self[index + 1]
    ## fill with the padding token
    label = np.full((self.max_len,), self.class_to_idx['PADDING'], dtype=np.int64)
    label_list = []
    for char in word:
      if char in self.class_to_idx:
        label_list.append(self.class_to_idx[char])
      else:
        label_list.append(self.class_to_idx['UNKNOWN'])
    ## add a stop token
    label_list = label_list + [self.class_to_idx['EOS']]
    assert len(label_list) <= self.max_len
    label[:len(label_list)] = np.array(label_list)
    if len(label) <= 0:
      return self[index + 1]
    
    # Label length
    label_len = len(label_list)

    # augmentation
    if self.use_aug:
      if self.use_abi_aug:
        aug_img = self.augment_abi(img)
      else:
        # augmentation
        aug_img = self.augmentor(np.asarray(img))
        aug_img = Image.fromarray(np.uint8(aug_img))
        aug_img = self.aug_transformer(aug_img)
      return aug_img, label, label_len
    else:
      assert self.transform is not None
      img = self.transform(img)
      # import pdb;pdb.set_trace()
      # import pdb;pdb.set_trace()
      # img = torch.as_tensor(
      #       np.ascontiguousarray(np.asarray(img).transpose(2, 0, 1))
      #   )/255
      # mean = torch.tensor([0.471, 0.448, 0.408]); std = torch.tensor([0.234, 0.239, 0.242])
      # mean = std = torch.tensor(0.5)
      # img = (img-mean)/std
      # img = (img-mean[:,None,None])/std[:,None,None]
      # tmp_img = img * 0.5 + 0.5
      # import torchvision
      # trans = torchvision.transforms.ToPILImage()
      # tmp_img = trans(tmp_img)
      # tmp_img.save('transform_vis/'+str(index)+'.jpg')
      return img, label, label_len
