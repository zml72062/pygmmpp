"""
dataloader.py - Load graph dataset to memory.
"""
from .data import Data
from .collate import collate
from .batch import Batch
from .dataset import Dataset
from typing import Optional, Any, Mapping, Sequence, Callable
import torch
import torch.utils.data


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 dataset: Dataset,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None,
                 drop_last: bool = False,
                 collator: Optional[Callable] = None,
                 **kwargs):
        """
        NOTE: The `Dataset` type can be any generic class which implements the
        index selection and concatenation interfaces.
        """
        super().__init__(dataset,
                         batch_size,
                         shuffle,
                         collate_fn=get_collate_fn(collator),
                         drop_last=drop_last,
                         **kwargs)


def get_collate_fn(collator: Optional[Callable]) -> Callable:
    if collator is None:
        collator = collate
    def collate_fn(data_list: Any):
        elem = data_list[0]
        if isinstance(elem, Data):
            return collator(Batch, data_list)
        elif isinstance(elem, torch.Tensor):
            return torch.utils.data.default_collate(data_list)
        elif isinstance(elem, float):
            return torch.tensor(data_list, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(data_list)
        elif isinstance(elem, str):
            return data_list
        elif isinstance(elem, Mapping):
            return {key: collate_fn([data[key] for data in data_list]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(collate_fn(s) for s in zip(*data_list)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [collate_fn(s) for s in zip(*data_list)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')
    return collate_fn
