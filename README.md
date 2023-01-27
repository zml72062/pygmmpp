# `PyG--++` --- A simple, PyTorch-based GNN library

`PyG--++` is a minimal library for GNNs based on PyTorch. It is a minimalist version of PyTorch Geometric, but some useful features are added. It is named `PyG--++` or PyG Minus Minus Plus Plus.

## Prerequisites

Packages `torch`, `torch_scatter`, `torch_sparse` should be installed.

## `Data` --- the basic data structure

For a graph with $n$ nodes and $m$ edges, we use $X\in \mathbb{R}^{n\times f_n}$ to encode its node features, $E\in \mathbb{R}^{m\times f_e}$ to encode its edge features, and $y\in \mathbb{R}^{1\times f_g}$ to encode its label. The adjacency matrix is denoted by $A\in \mathbb{R}^{n\times n}$.

We use the `Data` class, which wraps a `dict`, to store graph data. Each key-value pair of the `Data` class refers to a graph feature. 

## `Batch` --- support for batching

For efficient training on GPU, we need to combine a bag of graphs into a batch. 

The `Batch` class inherits from `Data`, and includes three extra fields: `batch`, `ptr` and `edge_slice`
* `batch` maps indices of nodes `i` to indices of graphs `batch[i]`
* `ptr` maps indices of graphs `i` to indices of nodes `ptr[i]`
* `edge_slice` maps indices of graphs `i` to indices of edges `edge_slice[i]`

The batching procedure can be applied to a bag of `Batch` objects in exactly the same way as `Data`. The result is again a `Batch` object. In this case, there will be two sets of `batch`, `ptr` and `edge_slice` vectors. We append an integral label `0, 1, ...` to distinguish the different sets of them. For example, `batch0` maps indices of nodes to indices of "individual" graphs, while `batch1` maps indices of nodes to indices of input batches.

The `torch_geometric` package offers an automatic batching for non-standard graph features (not `x`, `edge_index`, `edge_attr` or `y`), but we don't include it, and simply collect those features in a list.

## `Dataset` --- base class for datasets

We use a `Dataset` object, which simply wraps a `Batch` object, to store a graph dataset. When calling `__getitem__` on datasets, we return a graph from the dataset if the index is an integer, and return a "view" of the original dataset if the index is a slicing. This makes it zero-copy if we only want to split the dataset (into train / test, etc).

The `torch_geometric` package processes the datasets and the batches differently, which makes itself a less unified framework. Our profile test on real-world datasets proves that our treatment is almost as fast, but more elegant.

## `DataLoader` --- loading datasets

The `DataLoader` uses the `torch.utils.data.DataLoader` class, and is a simple wrapper class. 