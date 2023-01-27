"""
separate.py - Select a graph from a batch.
"""
from .data import Data
from .feature import *
import torch
import copy
from typing import Any


def get_ptr(obj: Any, idx: int):
    # get the top-level `ptr` vector
    ptr_list = torch.cat([obj.__dict__['ptr'+str(obj.batch_level)],
                          torch.tensor([obj.__inc__('ptr'+str(obj.batch_level))],
                                       dtype=int)])
    batch_level = obj.batch_level - 1

    # nodes in the graph has indices within [start, end)
    start, end = ptr_list[idx], ptr_list[idx+1]
    return batch_level, start, end


def get_edge_slice(obj: Any, idx: int):
    if 'edge_index' in obj.__dict__:
        # get the top-level `edge_slice` vector
        edge_slice_list = torch.cat([obj.__dict__['edge_slice'+str(obj.batch_level)],
                                     torch.tensor([obj.__inc__('edge_slice' +
                                                               str(obj.batch_level))],
                                                  dtype=int)])
        edge_start, edge_end = edge_slice_list[idx], edge_slice_list[idx+1]
        return edge_start, edge_end


def search(obj, start, end, edge_start, edge_end, idx):
    kwargs = {}
    for key in obj.keys():
        val = obj.__dict__[key]
        if isinstance(val, NodeFeature):
            kwargs[key] = obj.__dict__[key][start:end]
        elif isinstance(val, EdgeFeature):
            kwargs[key] = obj.__dict__[key][edge_start:edge_end]
        elif isinstance(val, GraphFeature):
            kwargs[key] = obj.__dict__[
                key][obj.batch0[start]:obj.batch0[end-1]+1]
        elif isinstance(val, EdgeIndex):
            kwargs[key] = obj.__dict__[key][:, edge_start:edge_end] - start
        elif (key not in obj.cat_dim_dict or key not in obj.inc_dict
              ) and key not in {'batch_level', 'inc_dict', 'cat_dim_dict'}:
            kwargs[key] = obj.__dict__[key][idx]
    return kwargs


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

    if 'edge_index' in obj.__dict__:
        # get the top-level `edge_slice` vector
        edge_slice_list = torch.cat([obj.__dict__['edge_slice'+str(obj.batch_level)],
                                     torch.tensor([obj.__inc__('edge_slice' +
                                                               str(obj.batch_level))],
                                                  dtype=int)])
        edge_start, edge_end = edge_slice_list[idx], edge_slice_list[idx+1]

    kwargs = {}
    for key in obj.keys():
        val = obj.__dict__[key]
        if isinstance(val, NodeFeature):
            kwargs[key] = obj.__dict__[key][start:end]
        elif isinstance(val, EdgeFeature):
            kwargs[key] = obj.__dict__[key][edge_start:edge_end]
        elif isinstance(val, GraphFeature):
            kwargs[key] = obj.__dict__[
                key][obj.batch0[start]:obj.batch0[end-1]+1]
        elif isinstance(val, EdgeIndex):
            kwargs[key] = obj.__dict__[key][:, edge_start:edge_end] - start
        elif (key not in obj.cat_dim_dict or key not in obj.inc_dict
              ) and key not in {'batch_level', 'inc_dict', 'cat_dim_dict'}:
            kwargs[key] = obj.__dict__[key][idx]

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

    kwargs['cat_dim_dict'] = cat_dim_dict
    kwargs['inc_dict'] = inc_dict
    kwargs['batch_level'] = batch_level

    # build the object
    if obj.batch_level == 0:
        return Data(**kwargs)
    else:
        out = cls(**kwargs)

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
