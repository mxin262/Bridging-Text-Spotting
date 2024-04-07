from __future__ import absolute_import

import numpy as np
import math
import string
from itertools import compress

import torch
import torch.nn.functional as F

from utils import to_torch, to_numpy

voc = list(string.printable[:-6])
voc.append('EOS')
voc.append('PADDING')
voc.append('UNKNOWN')


def _normalize_text(text):
  text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
  return text.lower()


def norm_multi_label_f_measure(preds, target):
  fs = []
  for pred, tgt in zip(preds, target):
    pred = pred[:94]
    tgt = tgt[:94]
    pred_char_list = list(compress(voc, pred))
    tgt_char_list = list(compress(voc, tgt))
    pred_str = ''.join(pred_char_list)
    tgt_str  = ''.join(tgt_char_list)

    norm_pred_str = _normalize_text(pred_str)
    norm_tgt_str = _normalize_text(tgt_str)

    num_right = 0.
    for norm_pred_char in norm_pred_str:
      if norm_pred_char in norm_tgt_str:
        num_right += 1
    p = num_right / (len(norm_pred_str) + 1e-5)
    r = num_right / (len(norm_tgt_str) + 1e-5)
    f = 2 * p * r / (p + r + 1e-5)
    fs.append(f)
  return sum(fs) / len(fs)

def multi_label_f_measure(logit, target, thres=0.5):
  """
    logit: [b, num_classes] unnormed logits
    target: [b, num_classes]
  """
  target = target.int()

  score = torch.sigmoid(logit)
  preds = (score > thres).int()

  return norm_multi_label_f_measure(preds, target)

  right_preds = (preds == target) * target
  precision = right_preds.sum(1) / (preds.sum(1) + 1e-5)
  recall = right_preds.sum(1) / (target.sum(1) + 1e-5)
  f = 2 * precision * recall / (precision + recall + 1e-5)

  # normalized f_measure without distinguishing upper and lower cases

  return f.mean().item()