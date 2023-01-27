"""
message_passing.py - Module for Message Passing GNN
"""
import torch
import torch.nn


class MessagePassing(torch.nn.Module):
    def message(**kwargs):
        raise NotImplementedError

    def update(aggr_out: torch.Tensor, **kwargs):
        raise NotImplementedError

    def __init__(self, aggr: str):
        pass

    def forward(self, edge_index, **kwargs):
        pass
