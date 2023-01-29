"""
pool.py - Graph pooling layers.
"""
import torch
import torch.nn as nn
import torch_scatter
from typing import Optional


class GlobalPool(nn.Module):
    """
    Global pooling layer.
    """

    def __init__(self, reduce: str = 'sum'):
        """
        `reduce` should be one of `'sum'`, `'mul'`, `'mean'`, `'min'`, or `'max'`
        """
        super().__init__()

        assert reduce in {'sum', 'mul', 'mean', 'min', 'max'}
        self.reduce = reduce

    def forward(self, x: torch.Tensor,
                batch: Optional[torch.Tensor] = None):
        if batch is None:
            return torch.__dict__[self.reduce](x, dim=0)
        return torch_scatter.scatter(x, batch, dim=0, reduce=self.reduce)
