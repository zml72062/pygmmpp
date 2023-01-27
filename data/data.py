"""
data.py - Data handling of graphs
"""
import torch
import torch_sparse
import copy
from .feature import *
from typing import Optional, Any, Tuple, Callable, Dict


class Data:
    """
    The `Data` class encodes graph data.

    A `Data` object can be seen as a `dict`, with its key-value pairs representing
    various graph features.
    """

    def __init__(self,
                 x: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.LongTensor] = None,
                 edge_attr: Optional[torch.Tensor] = None,
                 y: Optional[torch.Tensor] = None,
                 **kwargs):
        if x is not None:
            self.x = NodeFeature(x)
        if edge_index is not None:
            self.edge_index = EdgeIndex(edge_index)
        if edge_attr is not None:
            self.edge_attr = EdgeFeature(edge_attr)
        if y is not None:
            self.y = GraphFeature(y)

        self.batch_level = -1

        # maintain a dict for __cat_dim__() and __inc__() calls
        self.cat_dim_dict: Dict[str, int] = {}
        self.inc_dict: Dict[str, int] = {}

        self.__dict__.update(kwargs)

    def __setattr__(self, __name: str, __value: Any) -> None:
        self.__dict__[__name] = __value

    def __delattr__(self, __name: str) -> None:
        del self.__dict__[__name]

    def __repr__(self) -> str:
        feature_list = []
        for k in self.__dict__:
            v = self.__dict__[k]
            if isinstance(v, torch.Tensor):
                feature_list.append('='.join((k, repr(list(v.shape)))))
            elif k not in {'batch_level', 'inc_dict', 'cat_dim_dict'}:
                feature_list.append('='.join((k, repr(v))))
        return self.__class__.__name__ + '(' + ', '.join(feature_list) + ')'

    def keys(self):
        return self.__dict__.keys()

    def __iter__(self):
        return self.__dict__.__iter__()

    @property
    def num_nodes(self) -> int:
        if hasattr(self, 'x'):
            return self.x.shape[0]
        if hasattr(self, 'edge_index'):
            return int(self.edge_index.max() + 1)
        raise AttributeError("No attribute num_nodes!")

    @property
    def num_edges(self) -> int:
        if hasattr(self, 'edge_attr'):
            return self.edge_attr.shape[0]
        if hasattr(self, 'edge_index'):
            return self.edge_index.shape[1]
        raise AttributeError("No attribute num_edges!")

    @property
    def num_node_features(self) -> int:
        if hasattr(self, 'x'):
            return self.x.shape[1]
        return 0

    @property
    def num_edge_features(self) -> int:
        if hasattr(self, 'edge_attr'):
            return self.edge_attr.shape[1]
        return 0

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the adjacency matrix.
        """
        return (self.num_nodes, self.num_nodes)

    def is_undirected(self) -> bool:
        index_t, value_t = torch_sparse.transpose(index=self.edge_index,
                                                  value=torch.ones(
                                                      self.num_edges),
                                                  m=self.num_nodes, n=self.num_nodes)
        index, value = torch_sparse.coalesce(index=self.edge_index,
                                             value=torch.ones(self.num_edges),
                                             m=self.num_nodes, n=self.num_nodes)
        return bool(torch.all(index == index_t))

    def is_directed(self) -> bool:
        return not self.is_undirected()

    def has_self_loops(self) -> bool:
        src, tgt = self.edge_index
        return bool(torch.any(src - tgt == 0))

    def coalesce(self, reduce: str = 'add'):
        """
        If there are duplicate columns in `edge_index`, coalesce them via `reduce`.
        """
        edge_attr = self.edge_attr if hasattr(self, 'edge_attr') else None
        edge_index = self.edge_index if hasattr(self, 'edge_index') else None
        self.edge_index, self.edge_attr = torch_sparse.coalesce(
            edge_index, edge_attr, *self.shape, op=reduce)
        if self.edge_attr is None:
            del self.edge_attr

    def __cat_dim__(self, attr: str) -> int:
        """
        The `__cat_dim__` tells a dataloader the dimension on which a tensor-type
        graph feature `attr` should be concatenated.
        """
        val = self.__dict__[attr]
        if isinstance(val, (NodeFeature, EdgeFeature, GraphFeature)):
            return 0
        elif isinstance(val, EdgeIndex):
            return 1
        return self.cat_dim_dict[attr]

    def __inc__(self, attr: str) -> int:
        """
        The `__inc__` tells a dataloader the offset to add on a tensor-type graph
        feature `attr`.
        """
        val = self.__dict__[attr]
        if isinstance(val, (NodeFeature, EdgeFeature, GraphFeature)):
            return 0
        elif isinstance(val, EdgeIndex):
            return self.num_nodes
        return self.inc_dict[attr]

    def apply(self, func: Callable) -> "Data":
        """
        Apply `func` to all tensor-type graph features.
        """

        out = self.__new__(self.__class__)
        for k in self.__dict__:
            val = self.__dict__[k]
            if isinstance(val, torch.Tensor):
                out.__dict__[k] = func(val)
            else:
                out.__dict__[k] = val
        return out

    def to(self, *args, **kwargs) -> "Data":
        return self.apply(lambda x: x.to(*args, **kwargs))

    def clone(self) -> "Data":
        out = self.apply(lambda x: x.clone())
        for k in out.__dict__:
            val = out.__dict__[k]
            if not isinstance(val, torch.Tensor):
                out.__dict__[k] = copy.copy(val)
        return out

    def __set_tensor_attr__(self, name: str,
                            value: torch.Tensor,
                            type: str,
                            slicing: str,):
        """
        An extension to `__setattr__`, which allows auto-batching of new 
        tensor-type attributes.

        `type` should be one of `'node_feature'`, 
        """
