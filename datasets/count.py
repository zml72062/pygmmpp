import scipy.io as sio
import os.path as osp
import torch
import numpy as np
from ..data import Data, Dataset, Batch
from typing import List, Optional, Callable
from tqdm import tqdm

class CountDataset(Dataset):
    def __init__(self, root: str, target: Callable, 
                 target_name: str,
                 split: str='train',
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None, 
                 pre_filter: Optional[Callable] = None):
        """
        NOTE: `target` should be a function with the following prototype
        ```
        def target(edge_index: Union[torch.Tensor, np.ndarray],
                   num_nodes: int, num_edges: int)
                -> Union[torch.Tensor, np.ndarray]: ...
        ```
        This function should produce the node-level count of the substructure
        in a node-feature-like form. `target_name` can be the name of the 
        `target` function.
        """
        self.target = target
        self.target_name = target_name
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data_batch = torch.load(path)
        self.indices = torch.arange(len(self.data_batch))

    @property
    def raw_file_names(self):
        return 'randomgraph.mat'
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']
    
    def download(self):
        # Counting Dataset doesn't provide a download method
        pass

    def process(self):
        raw_data = sio.loadmat(self.raw_paths[0])
        slices, edge_indices, num_nodes = (
            raw_data['slices'][0],
            raw_data['edge_index'],
            raw_data['num_nodes'][0]
        )
        length = num_nodes.shape[0]

        data_list: List[Data] = []
        for i in tqdm(range(length)):
            edge_index = torch.from_numpy(
                    edge_indices[:, slices[i]:slices[i+1]])
            data_sample = Data(
                x=torch.ones((num_nodes[i], 1)),
                edge_index=edge_index
            )
            # generate substructure count at node-level
            count = self.target(
                    edge_index, num_nodes[i], edge_index.shape[1]
                )
            if isinstance(count, np.ndarray):
                count = torch.from_numpy(count)
            data_sample.__set_tensor_attr__(
                self.target_name, count, 'node_feature'
            )
            data_list.append(data_sample)
        
        for split in ['train', 'val', 'test']:
            idx = raw_data[f'{split}_idx'][0]

            split_data_list = [data_list[i] for i in idx]
            if self.pre_filter is not None:
                split_data_list = [data for data in split_data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                split_data_list = [self.pre_transform(data) for data in split_data_list]
        
            torch.save(Batch.from_data_list(split_data_list), 
                       osp.join(self.processed_dir, f'{split}.pt'))
