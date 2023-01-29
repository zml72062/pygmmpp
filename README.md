# `PyG--++` --- A simple, PyTorch-based GNN library

`PyG--++` is a minimal library for GNNs based on PyTorch. It is a minimalist version of PyTorch Geometric, but some useful features are added. It is named `PyG--++` or PyG Minus Minus Plus Plus.

## Prerequisites

Packages `torch`, `torch_scatter`, `torch_sparse` should be installed.

## `Data` --- the basic data structure

For a graph with $n$ nodes and $m$ edges, we use $X\in \mathbb{R}^{n\times f_n}$ to encode its node features, $E\in \mathbb{R}^{m\times f_e}$ to encode its edge features, and $y\in \mathbb{R}^{1\times f_g}$ to encode its label. The adjacency matrix is denoted by $A\in \mathbb{R}^{n\times n}$.

We use the `Data` class, which wraps a `dict`, to store graph data. Each key-value pair of the `Data` class refers to a graph feature. 

To make it easier to add new features into a graph, we introduced four special "feature classes":
* `node_feature`: like $X$, they are in the shape `(num_nodes * num_node_features)`, and has `cat_dim=0` and `inc=0`
* `edge_feature`: like $E$, they are in the shape `(num_edges * num_edge_features)`, and has `cat_dim=0` and `inc=0`
* `graph_feature`: like $y$, they are in the shape `(1 * num_graph_features)`, and has `cat_dim=0` and `inc=0`
* `edge_index`: like `edge_index`, they are in the shape `(2 * num_edges)`, and has `cat_dim=1` and `inc=num_nodes`

We store the keys that belong to those four classes in four distinct sets, and treat each of the class specially when calling `collate()` or `separate()`.

To conveniently add a new tensor-type feature to a `Data` object, we provide the `__set_tensor_attr__()` method, which is an extension to `__setattr__()`, by letting the caller decide whether the feature belongs to the above four "feature classes", or whether the feature needs auto-batching service. Moreover, this extension comes with a small overhead.

## `Batch` --- support for batching

For efficient training on GPU, we need to combine a bag of graphs into a batch. 

The `Batch` class inherits from `Data`, and includes three extra fields: `batch`, `ptr` and `edge_slice`
* `batch` maps indices of nodes `i` to indices of graphs `batch[i]`
* `ptr` maps indices of graphs `i` to indices of nodes `ptr[i]`
* `edge_slice` maps indices of graphs `i` to indices of edges `edge_slice[i]`

The batching procedure can be applied to a bag of `Batch` objects in exactly the same way as `Data`. The result is again a `Batch` object. In this case, there will be two sets of `batch`, `ptr` and `edge_slice` vectors. We append an integral label `0, 1, ...` to distinguish the different sets of them. For example, `batch0` maps indices of nodes to indices of "individual" graphs, while `batch1` maps indices of nodes to indices of input batches.

The `torch_geometric` package offers an automatic batching for non-standard graph features (not `x`, `edge_index`, `edge_attr` or `y`). We also include such mechanism: if an additional feature lies in any of the four specialized "feature classes", an automatic batching procedure is executed; otherwise, we simply collect them into a list. We believe our treatment is general enough to cover many interesting models.

## `Dataset` --- base class for datasets

We use a `Dataset` object, which simply wraps a `Batch` object, to store a graph dataset. When calling `__getitem__` on datasets, we return a graph from the dataset if the index is an integer, and return a "view" of the original dataset if the index is a slicing. This makes it zero-copy if we only want to split the dataset (into train / test, etc).

The `torch_geometric` package processes the datasets and the batches differently, which makes itself a less unified framework. Our profile test on real-world datasets proves that our treatment is a little (~1.3x time) slower, but more elegant.

## `DataLoader` --- loading datasets

The `DataLoader` uses the `torch.utils.data.DataLoader` class, and is a simple wrapper class. 

## `MessagePassing` --- message-passing layers

The `MessagePassing` class offers a handy way to define graph convolutional operators. To define a MPNN layer, one only needs to implement `message()` and `update()` methods (and `forward()`, optionally, though we have offered a default implementation).