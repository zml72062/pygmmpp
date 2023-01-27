"""
batch.py - Auto batching of graphs
"""
from .data import Data
from .collate import collate
from .separate import separate
import torch
import numpy
from typing import List, Union, Sequence


class Batch(Data):
    """
    A batch of graphs is stored as a **single** disconnected graph. A `Batch`
    object also implements `__getitem__()` which selects graph(s) from the batch.
    """

    def __init__(self, *args, **kwargs):
        """
        NOTE: All `Batch` objects must be instantiated by calling the `from_data_list`
        function.
        """
        super().__init__(*args, **kwargs)

    @classmethod
    def from_data_list(cls, data_list: List[Data]):
        """
        Make a batch from `data_list`.

        NOTE: The functionality of `collate` has been moved to `collate.py` since
        the `torch.utils.data.DataLoader` requires the `collate` function to support
        auto-batching of datasets.
        """
        return collate(cls, data_list)

    @property
    def num_graphs(self) -> int:
        return int(torch.max(self.__dict__['batch'+str(self.batch_level)])+1)

    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, idx: Union[
            int, slice, Sequence, numpy.ndarray, torch.Tensor]) -> Data:
        """
        Select the `idx`-th graph from the batch.

        NOTE: The functionality of `separate` has been moved to `separate.py`
        since the `pygmmpp.data.dataset` module needs the `separate` API.
        """
        # select a graph from the batch
        if isinstance(idx, int):
            return separate(Batch, self, idx)
        # select a bag of graphs from the batch
        elif isinstance(idx, slice):
            return Batch.from_data_list([self[l] for l
                                         in list(range(self.num_graphs))[idx]])
        else:
            return Batch.from_data_list([self[int(l)] for l in idx])
