import itertools
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler


class BatchSampler(torchBatchSampler):
    def __init__(self, *args, mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mosaic = mosaic

    def __iter__(self):
        for batch in super().__iter__():
            yield [(self.mosaic, idx) for idx in batch]


class InfiniteSampler(Sampler):
    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: Optional[int] = 0,
        rank=0,
        world_size=1,
    ):
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size
