"""
degree.py - Compute degree for graph data.
"""
import torch
from torch_scatter import scatter_add


def degree(index: torch.LongTensor, num_nodes: int) -> torch.Tensor:
    """
    Compute degree for the input index.

    `index` should be `edge_index[0]` or `edge_index[1]` of a graph. When it
    is `edge_index[0]`, computes out degree; when it is `edge_index[1]`,
    computes in degree.
    """
    src = torch.ones(index.shape[0], dtype=int)
    return scatter_add(src, index, dim_size=num_nodes)
