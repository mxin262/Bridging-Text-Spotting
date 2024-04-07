# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from turtle import Turtle
from typing import Iterable
from datetime import datetime
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from loss import SeqSimCLRLoss

def train_one_epoch(model: torch.nn.Module,
                    teacher_model: torch.nn.Module,
                    teacher_model_without_ddp: torch.nn.Module,
                    data_loader: Iterable, word_data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, momentum_schedule=None,
                    args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    use_aug = args.num_view > 1

    iters_per_epoch = len(data_loader)
    moco_m = args.moco_m

    # contrast loss_weight warmup
    if epoch == args.contrast_start_epoch:
      contrast_warmup_steps = min(args.contrast_warmup_steps, len(data_loader))
      contrast_loss_weights = np.linspace(0., args.loss_weight_contrast, contrast_warmup_steps)
      if contrast_warmup_steps < len(data_loader):
        contrast_loss_weights = np.hstack([contrast_loss_weights, np.ones(len(data_loader) - contrast_warmup_steps) * args.loss_weight_contrast])
    elif epoch > args.contrast_start_epoch:
      contrast_loss_weights = np.ones(len(data_loader)) * args.loss_weight_contrast
    else:
      contrast_loss_weights = np.zeros(len(data_loader))

    for step, (batch, text, text_lens) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # adjust momentum coefficient per iteration
        if args.use_moco_m_cos:
            moco_m = utils.adjust_moco_momentum(epoch + 1.0 * step / iters_per_epoch, args)
        else:
            moco_m = args.moco_m
        metric_logger.update(moco_m=moco_m)

        # pretrain
        if isinstance(batch, list):
            images, aug_images, bool_vis_masked_pos = batch
            images = images.to(device, non_blocking=True)
            aug_images = aug_images.to(device, non_blocking=True)
            bool_vis_masked_pos = bool_vis_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # prepare mim gt
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor((0.5, 0.5, 0.5)).to(device)[None, :, None, None]
            std = torch.as_tensor((0.5, 0.5, 0.5)).to(device)[None, :, None, None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            if use_aug:
                bool_vis_masked_pos = bool_vis_masked_pos.view(B, args.num_view, -1)
                
                # During mix training, aug_img has no mask and only ori_img has mask.
                if args.only_mim_on_ori_img:
                    bool_vis_masked_pos[:, 1, :].fill_(0)

                vis_labels = []
                for i in range(args.num_view):
                    vis_label = images_patch[bool_vis_masked_pos[:, i, :]].reshape(B, -1, C)
                    vis_labels.append(vis_label)
            else:
                vis_labels = [images_patch[bool_vis_masked_pos].reshape(B, -1, C)]

        with torch.cuda.amp.autocast():
            loss = 0.

            out_dict = model(images, aug_images, bool_vis_masked_pos, moco_m,
                             args.only_mim_on_ori_img)
                             
            # MoCo V3
            if 'contra_loss' in out_dict:
                contra_loss = out_dict['contra_loss']
                loss += (contra_loss * contrast_loss_weights[step])
                metric_logger.update(loss_contrast=contra_loss.item())

                if 'q1_acc1' in out_dict:
                    metric_logger.update(q1_acc1=out_dict['q1_acc1'])
                if 'q1_acc5' in out_dict:
                    metric_logger.update(q1_acc5=out_dict['q1_acc5'])
                if 'q2_acc1' in out_dict:
                    metric_logger.update(q2_acc1=out_dict['q2_acc1'])
                if 'q2_acc5' in out_dict:
                    metric_logger.update(q2_acc5=out_dict['q2_acc5'])

            # MIM
            if 'vis_out' in out_dict:
                vis_out = out_dict['vis_out']
                loss_pixel = 0.
                num_view = 1 if args.only_mim_on_ori_img else args.num_view

                for i in range(num_view):
                    loss_pixel += (1. / num_view) * F.mse_loss(vis_out[i], vis_labels[i], reduction='mean')
                
                loss += (loss_pixel * args.loss_weight_pixel)
                metric_logger.update(loss_pixel=loss_pixel.item())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
        
        if step >= 1 and step % (args.eval_freq * 10) == 0:
            utils.save_model(
                args=args, model=model, model_without_ddp=model.module,
                optimizer=optimizer,
                loss_scaler=loss_scaler, epoch="{0}_{1}".format(epoch, step),
            )
        
        # flush the screen info to disk_file.
        sys.stdout.flush()        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
