from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

def to_contiguous(tensor):
  if tensor.is_contiguous():
    return tensor
  else:
    return tensor.contiguous()

def _assert_no_grad(variable):
  assert not variable.requires_grad, \
    "nn criterions don't compute the gradient w.r.t. targets - please " \
    "mark these variables as not requiring gradients"

class SeqLabelSmoothingCrossEntropyLoss(nn.Module):
  def __init__(self, 
               weight=None,
               size_average=True,
               ignore_index=-100,
               sequence_normalize=False,
               sample_normalize=True,
               smoothing=0.1):
    super(SeqLabelSmoothingCrossEntropyLoss, self).__init__()
    self.weight = weight
    self.size_average = size_average
    self.ignore_index = ignore_index
    self.sequence_normalize = sequence_normalize
    self.sample_normalize = sample_normalize

    self.smoothing = smoothing
    self.confidence = 1. - smoothing

    assert (sequence_normalize and sample_normalize) == False

  def get_pad_mask(self, seq, seq_len):
      batch_size, max_len = seq.size()[:2]
      tmp1 = seq.new_zeros(max_len)
      tmp1[:max_len] = torch.arange(0, max_len, dtype=seq.dtype).unsqueeze(0)

      tmp1 = tmp1.expand(batch_size, max_len)
      tmp2 = seq_len.type(tmp1.type())
      tmp2 = tmp2.unsqueeze(1).expand(batch_size, max_len)
      # [N, max_len]
      mask = torch.lt(tmp1, tmp2)
      return mask

  def forward(self, input, target, length):
    _assert_no_grad(target)
    # length to mask
    batch_size, def_max_length = target.size(0), target.size(1)
    mask = self.get_pad_mask(target, length)
    input = to_contiguous(input).view(-1, input.size(2))
    logprobs = F.log_softmax(input, dim=1)
    target = to_contiguous(target).view(-1, 1)
    mask = to_contiguous(mask).view(-1, 1)
    nll_loss = - logprobs.gather(1, target.long()) * mask
    smooth_loss = - logprobs.mean(1) * mask
    loss = self.confidence * nll_loss + self.smoothing * smooth_loss

    loss = torch.sum(loss)
    if self.sequence_normalize:
      loss = loss / torch.sum(mask)
    if self.sample_normalize:
      loss = loss / batch_size
    
    return loss


if __name__ == '__main__':
  raw_crit = SeqLabelSmoothingCrossEntropyLoss(smoothing=0.)
  new_crit = SeqLabelSmoothingCrossEntropyLoss(smoothing=0.1)

  batch_size = 2
  real_max_len = 3
  num_classes = 10
  def_max_len = 6

  input = torch.randn(batch_size, real_max_len, num_classes)
  target = torch.IntTensor([[0,2,4,5,5,5], [1,3,5,5,5,5]])
  length = torch.IntTensor([3, 2])

  raw_loss = raw_crit(input, target, length)
  new_loss = new_crit(input, target, length)
  print(raw_loss, new_loss)