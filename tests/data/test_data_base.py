import sys
try:
    sys.path.append('.')
except:
    pass
import pygmmpp.data as mydata
import torch_geometric.data as pydata
import torch


def test_data_keys():
    torch.manual_seed(42)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 0, 2, 1, 2, 3, 4, 3, 5, 4, 5],
                               [1, 0, 2, 0, 2, 1, 4, 3, 5, 3, 5, 4]], dtype=int)
    edge_attr = torch.randn(12, 6)
    y = torch.randn(1, 3)
    name = 'graph01'
    myd = mydata.Data(x, edge_index, edge_attr, y, name=name)
    assert set(myd.keys()) == {'x', 'edge_index', 'edge_attr', 'name', 'y',
                               'cat_dim_dict', 'inc_dict', 'batch_level',
                               'node_feature_set', 'edge_feature_set',
                               'edge_index_set', 'graph_feature_set',
                               'require_slice_set'}
    del myd.name
    assert set(myd.keys()) == {'x', 'edge_index', 'edge_attr', 'y',
                               'cat_dim_dict', 'inc_dict', 'batch_level',
                               'node_feature_set', 'edge_feature_set',
                               'edge_index_set', 'graph_feature_set',
                               'require_slice_set'}
    myd.author = 'zml72062'
    assert set(myd.keys()) == {'x', 'edge_index', 'edge_attr', 'y', 'author',
                               'cat_dim_dict', 'inc_dict', 'batch_level',
                               'node_feature_set', 'edge_feature_set',
                               'edge_index_set', 'graph_feature_set',
                               'require_slice_set'}


def test_data_values():
    torch.manual_seed(42)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int)
    edge_attr = torch.randn(14, 6)
    y = torch.randn(1, 3)
    name = 'graph02'
    myd = mydata.Data(x, edge_index, edge_attr, y, name=name)
    pyd = pydata.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                      name=name)
    for k in {'x', 'edge_index', 'edge_attr', 'y', 'name'}:
        val = myd.__dict__[k]
        if isinstance(val, torch.Tensor):
            torch.testing.assert_close(val, pyd[k], rtol=1e-9, atol=1e-9)
        else:
            assert val == pyd[k]


def test_data_properties_1():
    torch.manual_seed(42)
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 2],
                               [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 2, 0]], dtype=int)
    data = mydata.Data(x, edge_index)
    assert data.num_nodes == 5
    assert data.num_edges == 12
    assert data.num_node_features == 10
    assert data.num_edge_features == 0
    assert data.is_undirected()
    assert not data.has_self_loops()
    assert data.shape == (5, 5)


def test_data_properties_2():
    torch.manual_seed(42)
    x = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1, 1, 1, 2, 3, 3, 3, 4, 0, 0, 2],
                               [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 2, 0]], dtype=int)
    edge_attr = torch.randn(12, 7)
    data = mydata.Data(x, edge_index, edge_attr)
    assert data.num_nodes == 5
    assert data.num_edges == 12
    assert data.num_node_features == 6
    assert data.num_edge_features == 7
    assert data.is_directed()
    assert data.has_self_loops()
