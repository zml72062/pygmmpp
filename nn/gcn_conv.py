"""
gcn_conv.py - Define the GCN convolutional layer.
"""

import torch
import torch.nn as nn
from .message_passing import MessagePassing
from ..utils.degree import degree
from ..utils.self_loop import add_self_loops_from_tensor


class GCNConv(MessagePassing):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 add_self_loops: bool = True,
                 bias: bool = True):
        super().__init__(aggr='sum')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.linear = nn.Linear(in_channels, out_channels, bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def message(self, edge_index: torch.LongTensor, x_j: torch.Tensor,
                num_nodes: int):
        src, tgt = edge_index
        deg = degree(src, num_nodes)
        fact = (deg[src] * deg[tgt]) ** (-0.5)
        fact[fact == float('inf')] = 0
        return fact.view(-1, 1) * x_j

    def update(self, aggr_out: torch.Tensor):
        return self.linear(aggr_out)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                num_nodes: int, **kwargs):
        if self.add_self_loops:
            edge_index, _ = add_self_loops_from_tensor(edge_index, num_nodes)
        return super().forward(x, edge_index, num_nodes=num_nodes, **kwargs)
