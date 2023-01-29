"""
self_loop.py - Adding or removing self loops in graph data.
"""
import torch
from ..data import Data
from typing import Optional, Tuple
import torch_sparse


def add_self_loops(data: Data, reduce: str = 'add') -> Data:
    out_data = data.clone()
    edge_index = data.edge_index if hasattr(data, 'edge_index') else None
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    out_data.edge_index, out_data.edge_attr = _add_self_loops(
        edge_index, data.num_nodes, edge_attr, reduce
    )
    if out_data.edge_attr is None:
        del out_data.edge_attr
    return out_data


def _add_self_loops(edge_index: torch.LongTensor, num_nodes: int,
                    edge_attr: Optional[torch.Tensor] = None,
                    reduce: str = 'add') -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
    out_edge_index = torch.cat((edge_index, torch.cat(
        (torch.arange(num_nodes).reshape(1, num_nodes),
         torch.arange(num_nodes).reshape(1, num_nodes)))), dim=1)
    if edge_attr is not None:
        num_edge_features = edge_attr.shape[1]
        out_edge_attr = torch.cat(
            (edge_attr, torch.zeros(num_nodes, num_edge_features)))
    else:
        out_edge_attr = None
    out_edge_index, out_edge_attr = torch_sparse.coalesce(
        out_edge_index, out_edge_attr, num_nodes, num_nodes, reduce
    )
    return out_edge_index, out_edge_attr


def remove_self_loops(data: Data) -> Data:
    out_data = data.clone()
    if hasattr(data, 'edge_index'):
        src, tgt = data.edge_index
        no_self_loop_idx = src != tgt
        out_data.edge_index = out_data.edge_index[:, no_self_loop_idx]
        if hasattr(data, 'edge_attr'):
            out_data.edge_attr = out_data.edge_attr[no_self_loop_idx]
    return out_data
