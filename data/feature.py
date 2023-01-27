"""
feature.py - Define the interface for node-level and edge-level features
"""
import torch


class EdgeFeature(torch.Tensor):
    """
    An edge feature is a tensor with shape `(num_edges * num_edge_features)`,
    and with `cat_dim=0` and `inc=0`.
    """
    pass


class NodeFeature(torch.Tensor):
    """
    A node feature is a tensor with shape `(num_nodes * num_node_features)`,
    and with `cat_dim=0` and `inc=0`.
    """
    pass


class EdgeIndex(torch.Tensor):
    """
    An edge index is a tensor of integers with shape `(2 * num_edges)`,
    and with `cat_dim=1` and `inc=num_nodes`.
    """
    pass


class GraphFeature(torch.Tensor):
    """
    A graph feature is a tensor with shape `(1 * num_graph_features)`, and
    with `cat_dim=0` and `inc=0`.
    """
