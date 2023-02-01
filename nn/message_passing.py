"""
message_passing.py - Module for Message Passing GNN
"""
import torch
import torch.nn
import torch_scatter
import inspect


class MessagePassing(torch.nn.Module):
    """
    The `MessagePassing` class defines a message-passing layer on graph data.
    The layer aggregates messages of neighboring nodes (defined by function
    `message()`), and uses the aggregated message to update the representation
    of the central node (defined by function `update()`).

    Any specific message-passing layer should implement the following methods:

    ```
    def message(self, **kwargs) -> Tensor: ...
    def update(self, aggr_out: Tensor, **kwargs) -> Tensor: ...
    ```
    """

    def message(self, **kwargs) -> torch.Tensor:
        """
        Defines how to calculate the messages to be aggregated and passed to
        the central node.
        """
        raise NotImplementedError

    def update(self, aggr_out: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Defines how the aggregated message should be used to update the 
        representation of the central node.
        """
        raise NotImplementedError

    def __init__(self, aggr: str,
                 flow: str = 'src_to_tgt',
                 node_feature_set: set[str] = {'x'}):
        """
        Args:

        `aggr` (`str`): the aggregation method, should be one of `'sum'`, `'mul'`, 
        `'mean'`, `'min'`, or `'max'`

        `flow` (`str`): the direction of message passing, should be one of 
        `'src_to_tgt'` or `'tgt_to_src'`

        `node_feature_set` (`set[str]`): the set of node features, expected to be
        from the `node_feature_set` field of a `Data` object
        """
        super().__init__()
        assert aggr in {'sum', 'mul', 'mean', 'min', 'max'
                        }, "Invalid aggregation method!"
        self.aggr = aggr

        assert flow in {'src_to_tgt', 'tgt_to_src'}
        self.flow = flow

        self.node_feature_set = node_feature_set

    def message_(self, x: torch.Tensor,
                 edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Wrapper function that calls `message()`.
        """
        # get input arguments for `message()` method
        message_in = inspect.signature(self.message).parameters.keys()

        # find the subset of `kwargs` that is a node feature
        node_feature = {k: v for (k, v) in kwargs.items()
                        if k in self.node_feature_set}
        node_feature.update({'x': x})

        if self.flow == 'src_to_tgt':
            src, tgt = edge_index
        else:
            tgt, src = edge_index

        # collect arguments for `message()` call
        message_kwargs = {k: v for (k, v) in kwargs.items()
                          if k not in self.node_feature_set}
        message_kwargs.update({
            k+'_j': v[src] for (k, v) in node_feature.items()
        })
        message_kwargs.update({
            k+'_i': v[tgt] for (k, v) in node_feature.items()
        })
        message_kwargs['edge_index'] = edge_index

        # call `message()`
        return self.message(**{
            k: message_kwargs[k] for k in message_in
        })

    def update_(self, aggr_out: torch.Tensor,
                x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Wrapper function that calls `update()`.
        """
        # get input arguments for `update()` method
        update_in = inspect.signature(self.update).parameters.keys()
        # collect arguments for `update()` call
        kwargs.update({'x': x, 'aggr_out': aggr_out})

        return self.update(**{
            k: kwargs[k] for k in update_in
        })

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor, **kwargs):

        if self.flow == 'src_to_tgt':
            src, tgt = edge_index
        else:
            tgt, src = edge_index

        messages = self.message_(x, edge_index, **kwargs)

        # execute aggregation
        aggr_out = torch_scatter.scatter(
            messages, tgt, dim=0, reduce=self.aggr
        )

        return self.update_(aggr_out, x, **kwargs)
