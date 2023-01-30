import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional


class Model(torch.nn.Module):
    """
    Class for building a standard GNN model.

    To define a GNN model, one should implement the method
    ```
    def init_layer(self, in_channels: int, out_channels: int, 
                   **kwargs) -> torch.nn.Module: ...
    ```

    Args:

    `in_channels (int)` - Size of input dimension

    `hidden_channels (int)` - Size of hidden dimension

    `num_layers (int)` - Number of MLP layers

    `out_channels (Optional[int])` - Size of output dimension; if `None`,
    set to `hidden_channels`

    `dropout (float)` - Dropout rate

    `residual (Optional[str])` - Whether to use jumping connection, can be `None`,
    `'add'`, or `'cat'`

    `batch_norm (bool)` - Whether to use batch normalization.

    `relu_first (bool)` - Whether to apply `nn.ReLU` before batch norm.
    """

    def init_layer(self, in_channels: int,
                   out_channels: int, **kwargs) -> torch.nn.Module:
        raise NotImplementedError

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 out_channels: Optional[int] = None,
                 dropout: float = 0.0,
                 residual: Optional[str] = None,
                 batch_norm: bool = True,
                 relu_first: bool = False,
                 **kwargs):
        super().__init__()

        if out_channels is None:
            out_channels = hidden_channels

        self.lins = torch.nn.ModuleList()
        self.lins.append(self.init_layer(
            in_channels, hidden_channels, **kwargs))
        for _ in range(num_layers-1):
            self.lins.append(self.init_layer(
                hidden_channels, hidden_channels, **kwargs))

        if residual != 'cat':
            self.lins.append(torch.nn.Linear(
                hidden_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(
                hidden_channels*num_layers, out_channels))

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            if batch_norm:
                self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
            else:
                self.norms.append(torch.nn.Identity())

        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.relu_first = relu_first
        self.residual = residual

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            reset_parameters(lin)
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(self, x: torch.Tensor):
        if self.residual is not None:
            emb_list = []

        for i in range(self.num_layers):
            x = self.lins[i](x)
            if self.relu_first:
                x = F.relu(x)
            x = self.norms[i](x)
            if not self.relu_first:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.residual is not None:
                emb_list.append(x)

        if self.residual is None:
            return self.lins[-1](x)
        elif self.residual == 'add':
            return self.lins[-1](sum(emb_list))
        elif self.residual == 'cat':
            return self.lins[-1](torch.cat(emb_list, dim=-1))

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_layers) + ')'


def reset_parameters(nn: torch.nn.Module):
    if hasattr(nn, 'reset_parameters'):
        nn.reset_parameters()
    elif hasattr(nn, 'children'):
        for child in nn.children():
            reset_parameters(child)


class MLP(Model):
    """
    Define a Multi-Layer Perceptron.
    """

    def init_layer(self, in_channels: int,
                   out_channels: int, bias: bool = True) -> torch.nn.Module:
        return torch.nn.Linear(in_channels, out_channels, bias)

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 out_channels: Optional[int] = None,
                 dropout: float = 0.0,
                 residual: Optional[str] = None,
                 batch_norm: bool = True,
                 bias: bool = True,
                 relu_first: bool = False,
                 **kwargs):
        super().__init__(in_channels,
                         hidden_channels,
                         num_layers,
                         out_channels,
                         dropout,
                         residual,
                         batch_norm,
                         relu_first,
                         bias=bias,
                         **kwargs)
