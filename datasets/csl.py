import logging
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import torch

from ..data import Data, Batch, Dataset
from ..data.sys import download_url, extract_zip
from ..utils import remove_self_loops_from_tensor


class CSL(Dataset):
    r"""CSL dataset.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    Stats:
            * - CSL
              - 150
              - ~41.0
              - ~164.0
              - 0
              - 10
    """
    url = 'https://www.dropbox.com/s/rnbkp5ubgk82ocu/CSL.zip?dl=1'

    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = 'CSL'
        """
        It is recommended to perform 5-fold cross validation with stratified
        sampling for CSL dataset.
        """
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data_batch = torch.load(self.processed_paths[0])
        self.indices = torch.arange(len(self.data_batch))

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'graphs_Kary_Deterministic_Graphs.pkl',
            'y_Kary_Deterministic_Graphs.pt'
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            adjs = pickle.load(f)

        ys = torch.load(self.raw_paths[1]).tolist()

        data_list = []
        for adj, y in zip(adjs, ys):
            row, col = torch.from_numpy(adj.row), torch.from_numpy(adj.col)
            edge_index = torch.stack([row, col], dim=0).to(torch.long)
            edge_index, _ = remove_self_loops_from_tensor(edge_index)
            data = Data(edge_index=edge_index, y=torch.tensor([y], dtype=torch.long)) 
            data.__set_tensor_attr__('n_nodes', 
                                     torch.tensor([adj.shape[0]], dtype=torch.long),
                                     'graph_feature')
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        torch.save(Batch.from_data_list(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
