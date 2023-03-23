"""
neighbor.py - Extracts k-hop neighbor of a given node.
"""
import torch


def k_hop_neighbor(node_idx: int, num_hops: int, edge_index: torch.LongTensor,
                   num_nodes: int) -> torch.BoolTensor:
    """
    Extracts up to k-hop shortest path distance neighbor of a given node.

    Returns a k-row mask tensor that selects nodes at each hop.
    """
    device = edge_index.device
    row, col = edge_index

    # mask of the nodes at each hop
    node_mask = torch.zeros(
        num_hops+1, num_nodes, dtype=torch.bool, device=device)

    edge_mask = torch.zeros(row.size(0), dtype=torch.bool, device=device)

    subsets = torch.tensor([node_idx], dtype=torch.long, device=device)
    # let `node_mask[hop]` be nodes with distance <= hop to node_idx
    for hop in range(num_hops+1):
        node_mask[hop:, subsets] = True
        edge_mask = node_mask[hop][row]
        subsets = col[edge_mask]

    return torch.diff(node_mask, dim=0)


def k_hop_edge_index(num_hops: int, edge_index: torch.LongTensor, num_nodes: int):
    """
    Extracts up to k-hop shortest path distance neighbors and gather them into
    `edge_index`-like tensors.

    Returns a dict containing all the additional features.
    """
    edge_index_dict = {'edge_index' +
                       str(hop): torch.zeros((2, 0), dtype=int)
                       for hop in range(2, num_hops+1)}

    for idx in range(num_nodes):
        index_base = torch.cat([torch.full((1, num_nodes), idx, dtype=int),
                               torch.arange(num_nodes, dtype=int).reshape(1, -1)])

        node_mask = k_hop_neighbor(idx, num_hops, edge_index, num_nodes)

        for hop in range(2, num_hops+1):
            edge_index_dict['edge_index'+str(hop)] = torch.cat([
                edge_index_dict['edge_index'+str(hop)],
                index_base[:, node_mask[hop-1]]
            ], dim=1)

    return edge_index_dict
