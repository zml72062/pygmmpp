"""
collate.py - Defining the default collating function
"""
from .data import Data
import torch
import copy
from typing import List


def collate(cls, data_list: List[Data]):
    """
    Collate a list `data_list` of graph data into a data batch.
    """
    assert len(data_list) > 0, "Empty data_list!"
    data_sample = data_list[0]

    batch = cls()

    batch.batch_level = data_sample.batch_level + 1

    for key in data_sample:
        value = data_sample.__dict__[key]

        # for tensor-type graph feature, concat them on the dimension
        # specified by `__cat_dim__()` and with increment `__inc__()`
        if isinstance(value, torch.Tensor):

            try:
                if data_sample.__inc__(key) != 0:
                    __inc__list = torch.tensor([data.__inc__(key)
                                                for data in data_list])
                    # calculate the increment of feature `key` for each element
                    increment = torch.cumsum(torch.concat(
                        (torch.tensor([0], dtype=int), __inc__list[:-1])
                    ), dim=0)

                    batch.__dict__[key] = torch.cat(
                        [data.__dict__[key] + increment[idx]
                            for idx, data in enumerate(data_list)],
                        dim=data_sample.__cat_dim__(key)
                    )
                else:
                    batch.__dict__[key] = torch.cat(
                        [data.__dict__[key] for data in data_list],
                        dim=data_sample.__cat_dim__(key)
                    )
            except KeyError:
                # If KeyError occurs, the tensor-type graph feature doesn't
                # appear in `inc_dict` and `cat_dim_dict`, thus shouldn't
                # be concatenated. Instead we maintain a list for them
                batch.__dict__[key] = [data.__dict__[key]
                                       for data in data_list]

            if key == 'edge_index':
                # add a top-level `batch` vector (with label `batch_level`)
                # for the batch

                # `batch` is like [0 0 ... 0 1 1 ... 1 ... n-1 n-1 ... n-1]
                # `batch[i] == j` means the node labeled i is in batch j
                batch.__dict__['batch'+str(batch.batch_level)] = torch.cat(
                    [torch.full((num_nodes,), idx)
                        for idx, num_nodes in enumerate(__inc__list)]
                )

                # When batching `Data`-like objects that already have
                # a `batch` field, we flatten them into a large batch
                # by concatenating the old `batch` vector
                batch.inc_dict['batch' +
                               str(batch.batch_level)] = len(data_list)
                batch.cat_dim_dict['batch'+str(batch.batch_level)] = 0

                # add a top-level `ptr` vector for the batch (with label
                # `batch_level`)

                # `ptr[i] == j` means batch i starts at node labeled j
                batch.__dict__['ptr'+str(batch.batch_level)] = increment

                batch.inc_dict['ptr'+str(batch.batch_level)] = len(
                    batch.__dict__['batch'+str(batch.batch_level)]
                )
                batch.cat_dim_dict['ptr'+str(batch.batch_level)] = 0

                # add a top-level slice vector for edges (with label
                # `batch_level`)
                edge_num_list = torch.tensor([data.__dict__[key].shape[
                    data.__cat_dim__(key)] for data in data_list])
                edge_slice = torch.cumsum(torch.concat(
                    (torch.tensor([0], dtype=int), edge_num_list[:-1])
                ), dim=0)

                batch.__dict__['edge_slice' +
                               str(batch.batch_level)] = edge_slice

                batch.inc_dict['edge_slice'+str(batch.batch_level)] = sum(
                    edge_num_list
                )
                batch.cat_dim_dict['edge_slice'+str(batch.batch_level)] = 0

        elif key == 'cat_dim_dict':
            for feature in value:
                batch.__dict__[key][feature] = value[feature]

        elif key == 'inc_dict':
            for feature in value:
                batch.__dict__[key][feature] = sum(
                    [data.__dict__[key][feature] for data in data_list]
                )
        elif '_set' in key:
            batch.__dict__[key] = copy.copy(data_sample.__dict__[key])

        # for non-tensor-type graph feature, maintain a list for them
        elif key != 'batch_level':
            batch.__dict__[key] = [data.__dict__[key]
                                   for data in data_list]
    return batch
