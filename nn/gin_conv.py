"""
gin_conv.py - Define the GIN and GINE convolutional layer.
"""

import torch
import torch.nn
import torch.nn.functional as F
from .message_passing import MessagePassing
from typing import Callable, Optional


class GINConv(MessagePassing):
    def __init__(self, nn: Callable,
                 eps: float = 0.0,
                 train_eps: bool = False):
        super().__init__(aggr='sum')
        self.nn = nn
        if train_eps:
            self.eps = torch.nn.parameter.Parameter(
                torch.tensor([eps], requires_grad=True)
            )
        else:
            self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.nn)

    def message(self, x_j: torch.Tensor):
        return x_j

    def update(self, aggr_out: torch.Tensor,
               x: torch.Tensor):
        return self.nn((1+self.eps) * x + aggr_out)


class GINEConv(MessagePassing):
    def __init__(self, nn: Callable,
                 eps: float = 0.0,
                 edge_dim: Optional[int] = None,
                 node_dim: Optional[int] = None,
                 train_eps: bool = False):
        super().__init__(aggr='sum')
        self.nn = nn
        if train_eps:
            self.eps = torch.nn.parameter.Parameter(
                torch.tensor([eps], requires_grad=True)
            )
        else:
            self.eps = eps

        if edge_dim is not None:
            assert node_dim is not None, "Must provide a node_dim "
            "if edge_dim is not None!"
            self.edge_dim = edge_dim
            self.lin = torch.nn.Linear(edge_dim, node_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.nn)
        if hasattr(self, 'edge_dim'):
            self.lin.reset_parameters()

    def message(self, x_j: torch.Tensor,
                edge_attr: torch.Tensor):
        if hasattr(self, 'edge_dim'):
            return F.relu(x_j + self.lin(edge_attr))
        else:
            return F.relu(x_j + edge_attr)

    def update(self, aggr_out: torch.Tensor,
               x: torch.Tensor):
        return self.nn((1+self.eps) * x + aggr_out)


def reset_parameters(nn: torch.nn.Module):
    if hasattr(nn, 'reset_parameters'):
        nn.reset_parameters()
    elif hasattr(nn, 'children'):
        for child in nn.children():
            reset_parameters(child)
