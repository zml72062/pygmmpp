"""
separate.py - Select a graph from a batch.
"""
from .data import Data
import torch
import copy
from typing import Any


def separate(cls, obj: Any, idx: int) -> Data:
    """
    Get the `idx`-th graph from `obj`, which is a batch of graphs.
    """
    # get the top-level `ptr` vector
    ptr_list = torch.cat([obj.__dict__['ptr'+str(obj.batch_level)],
                          torch.tensor([obj.__inc__('ptr'+str(obj.batch_level))],
                                       dtype=int)])
    batch_level = obj.batch_level - 1

    # nodes in the graph has indices within [start, end)
    start, end = ptr_list[idx], ptr_list[idx+1]

    x, y, edge_index, edge_attr = None, None, None, None
    if 'x' in obj.__dict__:
        # get a slice of node features
        x = obj.x[start:end]
    if 'y' in obj.__dict__:
        # get graph label
        y = obj.y[obj.batch0[start]:obj.batch0[end-1]+1]
    if 'edge_index' in obj.__dict__:
        # get the top-level `edge_slice` vector
        edge_slice_list = torch.cat([obj.__dict__['edge_slice'+str(obj.batch_level)],
                                     torch.tensor([obj.__inc__('edge_slice' +
                                                               str(obj.batch_level))],
                                                  dtype=int)])
        edge_start, edge_end = edge_slice_list[idx], edge_slice_list[idx+1]

        # get a slice of edge_index and decrement by correct value
        edge_index = obj.edge_index[:, edge_start:edge_end] - start

        if 'edge_attr' in obj.__dict__:
            # get a slice of edge features
            edge_attr = obj.edge_attr[edge_start:edge_end]

    # simply copy down `cat_dim_dict` and `inc_dict`
    # but delete top-level `batch`, `ptr` and `edge_slice` vectors
    cat_dim_dict = copy.copy(obj.cat_dim_dict)
    del cat_dim_dict['batch'+str(obj.batch_level)]
    del cat_dim_dict['ptr'+str(obj.batch_level)]
    del cat_dim_dict['edge_slice'+str(obj.batch_level)]

    inc_dict = copy.copy(obj.inc_dict)
    del inc_dict['batch'+str(obj.batch_level)]
    del inc_dict['ptr'+str(obj.batch_level)]
    del inc_dict['edge_slice'+str(obj.batch_level)]

    # for other features, select from a list
    # (corresponding to how they have been collated)
    kwargs = {}
    for key in obj.__dict__:
        if key not in {'x', 'edge_index', 'edge_attr', 'y', 'cat_dim_dict',
                       'inc_dict'} and not any(
                ('ptr' in key, 'batch' in key, 'edge_slice' in key)):
            kwargs[key] = obj.__dict__[key][idx]

    # build the object
    if obj.batch_level == 0:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                    cat_dim_dict=cat_dim_dict, inc_dict=inc_dict,
                    batch_level=batch_level, **kwargs)
    else:
        out = cls(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                  cat_dim_dict=cat_dim_dict, inc_dict=inc_dict,
                  batch_level=batch_level, **kwargs)

        # add non-top-level `batch`, `ptr` and `edge_slice` vectors
        # simply get a slice from `obj.batch`, `obj.ptr` and `obj.edge_slice`
        # and decrement by correct value
        for l in range(obj.batch_level):
            batch_l = obj.__dict__['batch'+str(l)][start:end]

            ptr_l = obj.__dict__['ptr'+str(l)][batch_l[0]:batch_l[-1]+1]
            edge_slice_l = obj.__dict__['edge_slice'+str(l)][
                batch_l[0]:batch_l[-1]+1]

            out.__dict__['batch'+str(l)] = batch_l - torch.min(batch_l)
            out.__dict__['ptr'+str(l)] = ptr_l - torch.min(ptr_l)
            out.__dict__['edge_slice' +
                         str(l)] = edge_slice_l - torch.min(edge_slice_l)

            # change their entries in `inc_dict`
            out.inc_dict['batch' +
                         str(l)] = int(torch.max(out.__dict__['batch'+str(l)])) + 1
            out.inc_dict['ptr' +
                         str(l)] = len(out.__dict__['batch'+str(l)])
            out.inc_dict['edge_slice' +
                         str(l)] = out.num_edges
        return out
