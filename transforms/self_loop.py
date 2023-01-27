"""
self_loop.py - Adding or removing self loops in graph data.
"""
import torch
from ..data import Data


def add_self_loops(data: Data) -> Data:
    out_data = data.clone()
    if hasattr(data, 'edge_index'):
        num_nodes = data.num_nodes
        out_data.edge_index = torch.cat((data.edge_index, torch.cat(
            (torch.arange(num_nodes).reshape(1, num_nodes),
                torch.arange(num_nodes).reshape(1, num_nodes)))), dim=1)
        if hasattr(data, 'edge_attr'):
            out_data.edge_attr = torch.cat(
                (data.edge_attr, torch.zeros(num_nodes, data.num_edge_features)))
        out_data.coalesce()
    return out_data


def remove_self_loops(data: Data) -> Data:
    out_data = data.clone()
    if hasattr(data, 'edge_index'):
        src, tgt = data.edge_index
        no_self_loop_idx = src != tgt
        out_data.edge_index = out_data.edge_index[:, no_self_loop_idx]
        if hasattr(data, 'edge_attr'):
            out_data.edge_attr = out_data.edge_attr[no_self_loop_idx]
    return out_data
