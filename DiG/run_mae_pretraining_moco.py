# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

from builtins import help
import argparse
import datetime
from email import generator
from email.policy import default
from unittest import defaultTestLoader
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import math
import sys
import random

from pathlib import Path

from timm.models import create_model
from optim_factory import create_optimizer

from dataset.datasets import build_pretraining_aloneimage_dataset
from dataset.dist_multisrc_batch_sampler import DistributedMultiSrcBatchWiseSampler
from dataset.concatdatasets import ConcatDataset
from engine_for_pretraining_moco import train_one_epoch
from utils.utils import NativeScalerWithGradNormCount as NativeScaler
from utils import utils
import modeling_pretrain_moco_mim_ori
from utils.logging import Logger


def get_args():
    parser = argparse.ArgumentParser('MAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--eval_freq', default=500, type=int)

    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--encoder_type', default='vit', type=str,
                        help='the type of feature encoder')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--text_mask_ratio', default=0.15, type=float,
                        help='ratio of the text tokens need be masked')
    parser.add_argument('--use_multiscale_mask', default=False, action='store_true',
                        help='whether use mask with different scales.')
    parser.add_argument('--aug_ratio', default=0., type=float,
                        help='ratio of using augmented patches.')
    parser.add_argument('--use_corner_mask', action='store_true', default=False)
    parser.add_argument('--corner_ratio', default=0., type=float,
                        help='the ratio of how many corners are used for masking.')
    parser.add_argument('--corner_prob', default=0., type=float,
                        help='the probability of original masking or corner-guided masking.')
    # parser.add_argument('--mask_scales', default=[1,2,4], type=int, nargs='+',
    #                     help='scales of used mask.')
    # parser.add_argument('--mask_ratios', default=[0.75], type=float, nargs='+',
    #                     help='ratios of each size mask.')
    parser.add_argument('--mask_scales', default=1, type=int,
                        help='scales of used mask.')
    parser.add_argument('--mask_ratios', default=0.75, type=float,
                        help='ratios of each size mask.')
    parser.add_argument('--num_view', default=1, type=int,
                        help='num_view masks are used for pretraining on a single image.')
    parser.add_argument('--use_abi_aug', action='store_true', default=False)
    parser.add_argument('--use_color_aug', action='store_true', default=False)
    parser.add_argument('--use_hard_sample', action='store_true', default=False)

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--input_h', default=32, type=int,
                        help='images input height for backbone')
    parser.add_argument('--input_w', default=128, type=int,
                        help='images input width for backbone')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
                        
    parser.add_argument('--normlize_target', default=False, type=bool,
                        help='normalized the target patch pixels')

    parser.add_argument('--num_mem_slots', default=0, type=int,
                        help='the number of memory bank to save some dataset distribution.')
    parser.add_argument('--use_mem_in_decoder', action='store_true', default=False,
                        help='whether using memory bank during decoding.')
    # EMA parameters
    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use teacher-student mode.')
    parser.add_argument('--momentum_teacher', type=float, default=0.996, help="""Base EMA
                        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
                        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--momentum_teacher_end', type=float, default=1.)
    parser.add_argument('--loss_feat_type', type=str, default='cossim')
    parser.add_argument('--loss_feat_beta', type=float, default=2, help='used in smoothl1 loss.')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")
    parser.add_argument('--fix_mask_token', action='store_true', default=False)

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--loss_win_size', type=int, default=3,
                        help='the size to calculate loss weights.')
    parser.add_argument('--use_loss_weight', action='store_true', default=False)
    parser.add_argument('--loss_weight_feat_align', type=float, default=0.,
                        help='the loss weight of feature alignment.')
    # Constrative learning
    parser.add_argument('--num_windows', type=int, default=5,
                        help='the numbers of split feature sequence.')
    parser.add_argument('--patchnet_name', type=str, default='regular',
                        help='the type of patchnet')
    parser.add_argument('--contrast_temperature', type=float, default=1,
                        help='the temperature used in contrastive learning.')
    parser.add_argument('--loss_weight_contrast', type=float, default=0.,
                        help='the loss weight of contrastive learning.')
    parser.add_argument('--contrast_warmup_steps', type=int, default=0,
                        help='steps to warmup mlm. After mlm_warmup_steps, becoming text_loss_weight.')
    parser.add_argument('--contrast_start_epoch', type=int, default=0,
                        help='epoch to start mlm. From this epoch on, start training mlm.')
    parser.add_argument('--loss_weight_consist', type=float, default=0.,
                        help='the loss weight of semantic consistency')
    parser.add_argument('--relation_window_size', type=str, default='8_32',
                        help='the window size to conduct relation distillation')
    parser.add_argument('--num_relation_heads', type=int, default=1,
                        help='the heads of relation')
    parser.add_argument('--relation_T', type=float, default=1.,
                        help='the temperature of relation')
    # Distillation learning
    parser.add_argument('--soft_label_type', type=str, default='none',
                        help='the type of distillation')
    parser.add_argument('--loss_weight_distill', type=float, default=0.,
                        help='the loss weight of distillation')
    parser.add_argument('--num_distribution', type=int, default=100000,
                        help='the number of instances on which to calculate the relation distribution.')
    parser.add_argument('--queue_size', type=int, default=65536,
                        help='the size of memory bank for contrastive learning.')
    parser.add_argument('--loss_weight_semgroup', type=float, default=0.,
                        help='the loss weight of semantic grouping')
    parser.add_argument('--distill_start_epoch', type=int, default=1,)
    parser.add_argument('--cluster_update_interval', type=str, default='step')
    ## moco specific configs:
    parser.add_argument('--moco_dim', default=256, type=int,
                        help='feature dimension (default: 256)')
    parser.add_argument('--moco_mlp_dim', default=4096, type=int,
                        help='hidden dimension in MLPs (default: 4096)')
    parser.add_argument('--moco_m', default=0.99, type=float,
                        help='moco momentum of updating momentum encoder (default: 0.99)')
    parser.add_argument('--moco_m_cos', action='store_true',
                        help='gradually increase moco momentum to 1 with a '
                            'half-cycle cosine schedule')
    parser.add_argument('--use_moco_m_cos', type=int, default=1,
                        help='whether use the moco_m_cos')
    parser.add_argument('--moco_t', default=1.0, type=float,
                        help='softmax temperature (default: 1.0)')
    parser.add_argument('--use_patch_transformer', action='store_true', default=False)
    parser.add_argument('--use_image_slice', action='store_true', default=False)
    parser.add_argument('--loss_weight_pixel', type=float, default=1.,
                        help='the loss weight of contrastive learning.')
    parser.add_argument('--only_mim_on_ori_img', action='store_true', default=False,
                        help='only using mim loss on the original image.'
                        'It is hard to convergence if use mim loss on the augmented image.')
    parser.add_argument('--alternately_training', action='store_true', default=False,
                        help='mim and moco v3 are alternately used for training.')
    parser.add_argument('--alternately_epoch_training', action='store_true', default=False,
                        help='alternately training over epoch.')
    parser.add_argument('--first_train_mim', action='store_true', default=False,
                        help='first train mim for half args.epochs')
    parser.add_argument('--use_mim', action='store_true', default=False)
    parser.add_argument('--use_moco', action='store_true', default=False)
    parser.add_argument('--attn_map_type', type=str, default='none')
    parser.add_argument('--recon_patch_scales', type=str, default='1')
    parser.add_argument('--num_target_layers', type=int, default=1)
    parser.add_argument('--corrupt_ops_ratios', type=str, default='1._0._0.')
    # rpp
    parser.add_argument('--loss_weight_pos', type=float, default=0.)
    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/train', nargs='+', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--train_url', default='/home/ma-user/modelarts/outputs/train_url_0/', type=str,)
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--voc_type', type=str, default='ALLCASES_SYMBOLS',
                        choices=['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'])
    parser.add_argument('--max_len', type=int, default=25)
    parser.add_argument('--num_samples', type=int, default=math.inf)
    ## word dataset parameters
    parser.add_argument('--mix_train_with_ctx', action='store_true', default=False)
    parser.add_argument('--ctx_path', type=str, default='/home/ymk-wh/workspace/datasets/text_recognition/synth_voc.pkl')
    parser.add_argument('--ctx_min_len', type=int, default=2)
    parser.add_argument('--ctx_max_len', type=int, default=25)
    parser.add_argument('--ctx_num_samples', type=int, default=math.inf)
    parser.add_argument('--ctx_nb_classes', type=int, default=97)
    parser.add_argument('--image_to_ctx_ratio', type=int, default=50)
    ## image-alone dataset parameters
    parser.add_argument('--image_alone_path', default='', nargs='+', type=str,
                        help='paths of only images')
    parser.add_argument('--mix_train_with_aloneimage', action='store_true', default=False)
    parser.add_argument('--aloneimage_num_samples', type=int, default=math.inf)
    ## image-text pair train with image or text branch
    parser.add_argument('--only_real_data_for_pretrain', action='store_true', default=False)
    parser.add_argument('--text_loss_weight', type=float, default=1.,
                        help='amp scaler will scale loss according to the loss value, \
                              now the vis and text losses are too different.')
    parser.add_argument('--vis_loss_weight', type=float, default=1.,
                        help='the same as the above.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        mlp_dim=args.moco_mlp_dim,
        dim=args.moco_dim,
        T=args.moco_t,
        num_windows=args.num_windows,
        encoder_type=args.encoder_type,
        queue_size=args.queue_size,
        patchnet_name=args.patchnet_name,
    )

    return model


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    sys.stdout = Logger(os.path.join(args.output_dir, 'screen.txt'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_h // patch_size[0], args.input_w // patch_size[1])
    args.patch_size = patch_size

    # EMA training
    if args.use_ema:
        # teacher model
        teacher_model = get_model(args)
        teacher_model.to(device)
        if args.distributed:
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu], find_unused_parameters=False)
            teacher_model_without_ddp = teacher_model.module
        # teacher and student start with the same weights
        teacher_model_without_ddp.load_state_dict(model.state_dict(), strict=False)
        for p in teacher_model.parameters():
            p.requires_grad = False
    else:
        teacher_model = None
        teacher_model_without_ddp = None

    # get dataset
    dataset_train = build_pretraining_aloneimage_dataset(args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # Within dataloader, mask is random generated. However, each worker is initialized with the same seed.
    # What's more, the random seed is repeated every epoch.
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=seed_worker,
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.lr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, args.momentum_teacher_end, args.epochs, num_training_steps_per_epoch
    )

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model, teacher_model, teacher_model_without_ddp, data_loader_train, None,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            momentum_schedule=momentum_schedule,
            patch_size=patch_size[0],
            normlize_target=args.normlize_target,
            args=args,
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        os.makedirs(opts.output_dir, exist_ok=True)
    main(opts)
