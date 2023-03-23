import sys
try:
    sys.path.append('.')
except:
    pass
import torch
from pygmmpp.datasets.tu_dataset import TUDataset
from pygmmpp.data import Data
from pygmmpp.utils.neighbor import k_hop_edge_index
from typing import Callable, Dict

def transform(max_hop: int) -> Callable:
    def k_hop_neighbor_injection(data: Data) -> Data:
        edge_index_dict: Dict[str, torch.Tensor] = k_hop_edge_index(
            max_hop, data.edge_index, data.num_nodes
        )
        # keep raw data immutable
        data_ = data.clone()
        for (key, val) in edge_index_dict.items():
            data_.__set_tensor_attr__(key, val, 'edge_index', slicing=True)
        return data_
    return k_hop_neighbor_injection

def test_transform_mutag():
    data_raw = TUDataset(root='data/TU/raw', name='MUTAG')
    data_proc = TUDataset(root='data/TU/processed',
                          name='MUTAG', pre_transform=transform(3))
    torch.testing.assert_close(
        data_proc[0].edge_index2, torch.tensor(
            [[0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7,
              7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12,
              12, 13, 13, 13, 13, 14, 14, 15, 15, 16, 16],
             [2, 4, 3, 5, 0, 4, 9, 1, 5, 6, 8, 10, 0, 2, 7, 9, 1, 3, 6, 3, 5, 8, 4,
                9, 13, 3, 6, 10, 12, 2, 4, 7, 11, 13, 3, 8, 12, 9, 13, 14, 8, 10, 15,
                16, 7, 9, 11, 14, 11, 13, 12, 16, 12, 15]]
        ), rtol=1e-9, atol=1e-9
    )
    torch.testing.assert_close(
        data_proc[0].edge_index3, torch.tensor(
            [[0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7,
              7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11,
                12, 12, 13, 13, 13, 13, 13, 14, 14, 15, 15, 16, 16],
             [3, 6, 4, 9, 5, 6, 8, 10, 0, 7, 11, 13, 1, 8, 10, 2, 7, 9, 0, 2, 9,
              13, 3, 5, 10, 12, 2, 4, 11, 14, 1, 5, 6, 12, 2, 4, 7, 13, 14, 3, 8,
              15, 16, 7, 9, 3, 6, 10, 15, 16, 8, 10, 11, 13, 11, 13]]
        ), rtol=1e-9, atol=1e-9
    )
    # test `__inc__` correct
    torch.testing.assert_close(
        data_proc.data_batch.edge_index[:, 38:66] -
        data_proc[1].edge_index - 17,
        torch.zeros(2, 28, dtype=int), rtol=1e-9, atol=1e-9
    )
    torch.testing.assert_close(
        data_proc.data_batch.edge_index2[:, 54:92] -
        data_proc[1].edge_index2 - 17,
        torch.zeros(2, 38, dtype=int), rtol=1e-9, atol=1e-9
    )
    torch.testing.assert_close(
        data_proc.data_batch.edge_index3[:, 56:94] -
        data_proc[1].edge_index3 - 17,
        torch.zeros(2, 38, dtype=int), rtol=1e-9, atol=1e-9
    )
