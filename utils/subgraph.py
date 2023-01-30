"""
subgraph.py - Extract subgraphs from graph data.
"""

import torch
from typing import Union, Optional, List, Tuple


def subgraph(subset: Union[torch.LongTensor, List[int], torch.BoolTensor],
             edge_index: torch.LongTensor, num_nodes: int,
             edge_attr: Optional[torch.Tensor] = None,
             relabel_nodes: bool = False) -> Tuple[
    torch.LongTensor, Optional[torch.Tensor], torch.BoolTensor
]:
    """
    Extracts subgraph induced by `subset`.

    `subset` can be a subset of nodes, or boolean mask of a subset of nodes.

    When `relabel_nodes` is set to `True`, node indices of the output `edge_index`
    will be relabeled from 0, 1, ...; otherwise, the original node indices are
    kept.

    Returns:

    (`edge_index`, `edge_attr`, `edge_mask`) - The `edge_index` and `edge_attr` fields
    of the subgraph, along with a mask vector that selects edges in the subgraph.
    """

    device = edge_index.device

    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    if subset.dtype == torch.bool:
        node_mask: torch.BoolTensor = subset
    else:
        node_mask: torch.BoolTensor = torch.zeros(
            num_nodes, dtype=torch.bool, device=device)
        node_mask[subset] = True

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]

    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
                               device=device)
        node_idx[node_mask] = torch.arange(
            node_mask.sum().item(), device=device)
        edge_index = node_idx[edge_index]

    return edge_index, edge_attr, edge_mask


def k_hop_subgraph(node_idx: Union[torch.LongTensor, List[int], int],
                   num_hops: int, edge_index: torch.LongTensor,
                   num_nodes: int, relabel_nodes: bool = False) -> Tuple[
    torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.BoolTensor
]:
    """
    Extract k-hop subgraph centered by `node_idx`.

    Returns:

    `subset` - nodes involved in the k-hop subgraph

    `edge_index` - connectivity of the k-hop subgraph

    `inv` - index of the original `node_idx` in `subset`

    `edge_mask` - mask that indicates edges involved in the k-hop subgraph
    """
    row, col = edge_index

    node_mask = torch.zeros(num_nodes, dtype=torch.bool,
                            device=edge_index.device)
    edge_mask = torch.zeros(
        row.size(0), dtype=torch.bool, device=edge_index.device)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        edge_mask = node_mask[row]
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    edge_index, _, edge_mask = subgraph(subset, edge_index, num_nodes,
                                        relabel_nodes=relabel_nodes)

    return subset, edge_index, inv, edge_mask
