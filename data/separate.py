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
    kwargs['batch_level'] = obj.batch_level - 1

    for key in {'node_feature_set', 'edge_feature_set',
                'edge_index_set', 'graph_feature_set',
                'require_slice_set'}:
        kwargs[key] = copy.copy(obj.__dict__[key])

    # simply copy down `cat_dim_dict` and `inc_dict`, ...
    cat_dim_dict = copy.copy(obj.cat_dim_dict)
    inc_dict = copy.copy(obj.inc_dict)
    kwargs['cat_dim_dict'] = cat_dim_dict
    kwargs['inc_dict'] = inc_dict

    # those keys correspond to features that have been collated
    # into a list
    other_keys = set(obj.keys()).difference(set(kwargs.keys()),
                                            obj.node_feature_set,
                                            obj.edge_feature_set,
                                            obj.edge_index_set,
                                            obj.graph_feature_set,
                                            obj.require_slice_set,
                                            set(cat_dim_dict.keys()),
                                            set(inc_dict.keys()))

    # ..., but delete top-level `batch`, `ptr` and `edge_slice` vectors
    del kwargs['cat_dim_dict']['batch'+str(obj.batch_level)]
    del kwargs['cat_dim_dict']['ptr'+str(obj.batch_level)]
    del kwargs['cat_dim_dict']['edge_slice'+str(obj.batch_level)]

    del kwargs['inc_dict']['batch'+str(obj.batch_level)]
    del kwargs['inc_dict']['ptr'+str(obj.batch_level)]
    del kwargs['inc_dict']['edge_slice'+str(obj.batch_level)]

    for key in obj.require_slice_set:
        del kwargs['cat_dim_dict'][key+'_slice'+str(obj.batch_level)]
        del kwargs['inc_dict'][key+'_slice'+str(obj.batch_level)]

    for key in obj.node_feature_set:
        kwargs[key] = obj.__dict__[key][start:end]
    for key in obj.edge_feature_set:
        kwargs[key] = obj.__dict__[key][edge_start:edge_end]
    for key in obj.graph_feature_set:
        kwargs[key] = obj.__dict__[
            key][obj.batch0[start]:obj.batch0[end-1]+1]

    if 'edge_index' in obj.__dict__:
        kwargs['edge_index'] = obj.edge_index[:, edge_start:edge_end]

    for key in obj.require_slice_set:
        val = obj.__dict__[key]
        slicing = len(val.shape) * [slice(None)]
        key_slice_list = torch.cat([obj.__dict__[key+'_slice'+str(obj.batch_level)],
                                    torch.tensor([obj.__inc__(key+'_slice' +
                                                              str(obj.batch_level))],
                                                 dtype=int)])
        key_start, key_end = key_slice_list[idx], key_slice_list[idx+1]
        slicing[obj.__cat_dim__(key)] = slice(key_start, key_end)
        kwargs[key] = val[slicing]

    for key in obj.edge_index_set:
        kwargs[key] = kwargs[key] - start
    for key in other_keys:
        kwargs[key] = obj.__dict__[key][idx]

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

            for key in obj.require_slice_set:
                key_slice_l = obj.__dict__[key+'_slice'+str(l)][
                    batch_l[0]:batch_l[-1]+1]
                out.__dict__[key+'_slice' +
                             str(l)] = key_slice_l - torch.min(key_slice_l)
                out.inc_dict[key+'_slice'+str(l)] = out.__dict__[key].shape[
                    out.__cat_dim__(key)
                ]
        return out
