# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from builtins import getattr
import os
import torch
import math

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from timm.data import create_transform

from masking_generator import RandomMaskingGenerator
from dataset.dataset_folder import ImageFolder
from dataset.dataset_lmdb import ImageLmdb
from dataset.dataset_image import AloneImageLmdb
from dataset.concatdatasets import ConcatDataset


class DataAugmentationForMAE(object):
    def __init__(self, args):
        mean = std = 0.5

        self.transforms = transforms.Compose([
            transforms.Resize((args.input_h, args.input_w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.image_masked_position_generator = RandomMaskingGenerator(args.window_size, args.mask_ratio, num_view=args.num_view,)

    def __call__(self, image):
        return self.transforms(image), self.image_masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transforms)
        repr += "  Masked position generator = %s,\n" % str(self.image_masked_position_generator)
        repr += ")"
        return repr

# only image
def build_pretraining_aloneimage_dataset(args):
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))
    if isinstance(args.image_alone_path, list):
        dataset_list = []
        for image_alone_path in args.image_alone_path:
            dataset = AloneImageLmdb(image_alone_path, args.voc_type, 
                                     args.max_len, args.aloneimage_num_samples, transform=transform, use_aug=args.num_view>1.)
            dataset_list.append(dataset)
        return ConcatDataset(dataset_list)
    else:
        return AloneImageLmdb(args.image_alone_path, args.voc_type, args.max_len,
                              args.aloneimage_num_samples, transform=transform, use_aug=args.num_view>1.)


def build_dataset(is_train, args, use_mim_mask=False):
    use_abi_aug = getattr(args, 'use_abi_aug', False)

    # transform = build_transform(is_train, args)
    if use_mim_mask:
        transform = DataAugmentationForMAE(args)
    else:
        transform = RegularTransform(is_train, args)
    num_view = getattr(args, 'num_view', 1)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    elif isinstance(transform, RegularTransform) or isinstance(transform, DataAugmentationForMAE):
        for t in transform.transforms.transforms:
            print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    elif args.data_set == 'image_lmdb':
        root = args.data_path if is_train else args.eval_data_path
        if isinstance(root, list):
            dataset_list = []
            for data_path in root:
                dataset = ImageLmdb(data_path, args.voc_type, args.max_len,
                    args.num_samples if is_train else math.inf, transform=transform,
                    use_aug=(num_view>1. and is_train), use_abi_aug=use_abi_aug)
                dataset_list.append(dataset)
            dataset = ConcatDataset(dataset_list)
        else:
            dataset = ImageLmdb(root, args.voc_type, args.max_len,
                        args.num_samples if is_train else math.inf, transform=transform,
                        use_aug=(num_view>1. and is_train), use_abi_aug=use_abi_aug)
        nb_classes = len(dataset.classes)
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    mean = std = 0.6
    t = []
    t.append(transforms.Resize((args.input_h, args.input_w), interpolation=3))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class RegularTransform(object):
    def __init__(self, is_train, args):
        mean = std = 0.5
        # mean = [0.471, 0.448, 0.408]; std = [0.234, 0.239, 0.242]
        self.transforms = transforms.Compose([
            transforms.Resize((args.input_h, args.input_w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

    def __call__(self, image):
        return self.transforms(image)