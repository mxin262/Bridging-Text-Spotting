import math
from typing import TypeVar, Optional, Iterator
import itertools
import random

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)


class DistributedMultiSrcBatchWiseSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 batch_size: int = 0) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        # args for multi-datasets
        self.batch_size = batch_size
        self.num_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])
        # for each dataset, drop_last according to both `num_replicas` and `batch_size`
        # total_sizes is the dropped_last size
        # start_idxs is the start_idx of each dataset
        self.total_sizes = []
        self.cumulative_sizes = self.dataset.cumulative_sizes
        for cur_dataset in dataset.datasets:
            # drop_last for distribution (num_replicas)
            # If the dataset length is evenly divisible by # of replicas, then there
            # is no need to drop any data, since the dataset will be split equally.
            if len(cur_dataset) % self.num_replicas != 0:
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                cur_num_samples = math.ceil((len(cur_dataset) - self.num_replicas) / self.num_replicas)
            else:
                cur_num_samples = math.ceil(len(cur_dataset) / self.num_replicas)
            # drop_last for batch training (batch_size)
            if cur_num_samples % self.batch_size != 0:
                cur_num_samples = math.ceil((cur_num_samples - self.batch_size) / self.batch_size) * self.batch_size
            cur_total_size = cur_num_samples * self.num_replicas
            self.total_sizes.append(cur_total_size)
        self.total_size = sum(self.total_sizes)
        assert self.total_size % (self.batch_size * self.num_replicas) == 0
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = []
            start_idx = 0
            for dataset_idx, cur_total_size in enumerate(self.total_sizes):
                cur_indices = torch.randperm(cur_total_size, generator=g)
                cur_indices = (start_idx + cur_indices).tolist()
                indices += cur_indices
                start_idx = self.cumulative_sizes[dataset_idx]
        else:
            indices = list(range(self.total_size))
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # shuffle batches
        # if self.shuffle:
        # if False:
        if True:
            random.seed(self.seed + self.epoch)
            blocks = [indices[x:x+self.batch_size] for x in range(0, len(indices), self.batch_size)]
            random.shuffle(blocks)
            indices = list(itertools.chain.from_iterable(blocks))
        assert len(indices) == self.total_size // self.num_replicas

        return iter(indices)

    def __len__(self) -> int:
        return self.total_size // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch