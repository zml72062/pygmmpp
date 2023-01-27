"""
dataset.py - Defining the interface of graph datasets
"""
import torch
import torch.utils.data
import os.path as osp
import warnings
import sys
import re
import copy
import numpy
from typing import Any, List, Sequence, Union, Tuple, Optional, Callable
from .sys import makedirs
from .data import Data
from .batch import Batch


class Dataset(torch.utils.data.Dataset):
    """
    A graph dataset is stored as a batch of graphs, with necessary metadata. If there
    are extra data along with `Data` objects, the dataset should be stored as a batch
    of graphs along with a list of extra data.

    Any specific graph dataset should implement the following method:

    ```
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]: ...
    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]: ...
    def download(self): ...
    def process(self): ...
    ```
    """
    data_batch: Union[Batch, Tuple[Batch, Any]]
    # We use a vector `indices` to record the indices of actual data in the
    # dataset. This avoids unnecessary re-processing when slicing datasets.
    indices: torch.Tensor

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        """ File names for the raw data. """
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """ File names for the processed data. """
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def __init__(self, root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        """
        Args:
        root : Root directory where the dataset should be saved.
        transform : A function/transform that takes in a `Data` object and returns
            a transformed version. The data object will be transformed before 
            every access.
        pre_transform : A function/transform that takes in a `Data` object and 
            returns a transformed version. The data object will be transformed 
            before being saved to disk.
        pre_filter : A function that takes in a `Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset.
        """
        super().__init__()

        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))

        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        # Instantiating a subclass of `Dataset`
        if self.download.__qualname__.split('.')[0] != 'Dataset':
            self._download()

        if self.process.__qualname__.split('.')[0] != 'Dataset':
            self._process()

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_paths(self) -> List[str]:
        """ Full paths for the raw data. """
        files = to_list(self.raw_file_names)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self) -> List[str]:
        """ Full paths for the processed data. """
        files = to_list(self.processed_file_names)
        return [osp.join(self.processed_dir, f) for f in files]

    def _get_batch(self) -> Batch:
        if isinstance(self.data_batch, Batch):
            return self.data_batch
        return self.data_batch[0]

    @property
    def num_node_features(self) -> int:
        return self._get_batch().num_node_features

    @property
    def num_edge_features(self) -> int:
        return self._get_batch().num_edge_features

    def _download(self):
        """
        Wrapper function for downloading a dataset.
        """
        # If raw data files exist, do not re-download
        if files_exist(self.raw_paths):
            return

        makedirs(self.raw_dir)
        # Call the `download` method implemented by subclass
        self.download()

    def _process(self):
        """
        Wrapper function for pre-processing a dataset.
        """
        f = osp.join(self.processed_dir, 'pre_transform.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first")

        f = osp.join(self.processed_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, make sure to "
                "delete '{self.processed_dir}' first")

        # If processed data files exist, do not re-process
        if files_exist(self.processed_paths):
            return

        print('Processing...', file=sys.stderr)

        makedirs(self.processed_dir)
        # Call the `process` method implemented by subclass
        self.process()

        path = osp.join(self.processed_dir, 'pre_transform.pt')
        torch.save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter.pt')
        torch.save(_repr(self.pre_filter), path)

        print('Done!', file=sys.stderr)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: Union[
            int, slice, Sequence,
            numpy.ndarray, torch.Tensor]) -> Union[Data, Tuple[Data, Any]]:
        # when `idx` is an integer, we select one graph from the dataset
        if isinstance(idx, int):
            if isinstance(self.data_batch, Batch):
                data = self.data_batch[int(self.indices[idx])]
                return data if self.transform is None else self.transform(data)
            else:
                data = self.data_batch[0][int(self.indices[idx])]
                return (data if self.transform is None else self.transform(data)), tuple(
                    (field[int(self.indices[idx])] for field in self.data_batch[1]))
        # when `idx` is a slice object or a sequence,
        # returns a shallow copy of the dataset
        # with `indices` restricted to the selected part
        else:
            full = copy.copy(self)
            full.indices = self.indices[idx]
            return full

    def shuffle(self):
        # returns a shallow copy of the dataset
        # with `indices` vector shuffled
        full = copy.copy(self)
        full.indices = full.indices[torch.randperm(len(self))]
        return full

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.__class__.__name__}({arg_repr})'


###################  Helper functions  #################

def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def _repr(obj: Any) -> str:
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())
