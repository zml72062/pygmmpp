import sys
try:
    sys.path.append('.')
except:
    pass
import pygmmpp.data as mydata
from pygmmpp.utils.self_loop import *
from pygmmpp.utils.degree import *
import torch


def test_utils_add_self_loop1():
    torch.manual_seed(142)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int)
    edge_attr = torch.randn(14, 6)
    y = torch.randn(1, 3)
    name = 'graph02'
    myd = mydata.Data(x, edge_index, edge_attr, y, name=name)

    added = add_self_loops(myd)
    torch.testing.assert_close(added.x, x, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(added.edge_index, torch.tensor(
        [[0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5],
         [0, 1, 3, 0, 1, 2, 4, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4, 5, 4, 5]]), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(added.edge_attr, torch.cat([edge_attr, torch.zeros(6, 6)])[
        [14, 0, 10, 1, 15, 2, 13, 3, 16, 4, 11, 5, 17, 6, 12, 7, 18, 8, 9, 19]
    ], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(added.y, y, rtol=1e-9, atol=1e-9)


def test_utils_add_self_loop2():
    torch.manual_seed(452)
    x = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1, 1, 1, 2, 3, 3, 3, 4, 0, 0, 2],
                               [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 2, 0]], dtype=int)
    edge_attr = torch.randn(12, 7)
    data = mydata.Data(x, edge_index, edge_attr)

    added = add_self_loops(data)
    torch.testing.assert_close(added.x, x, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(added.edge_index, torch.tensor(
        [[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
         [0, 1, 2, 4, 0, 1, 2, 0, 2, 3, 2, 3, 4, 0, 4]]
    ), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(added.edge_attr, torch.cat([edge_attr, torch.zeros(3, 7)])[
        [12, 0, 10, 9, 1, 3, 2, 11, 13, 4, 5, 7, 6, 8, 14]
    ], rtol=1e-9, atol=1e-9)


def test_utils_remove_self_loop():
    torch.manual_seed(352)
    x = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1, 1, 1, 2, 3, 3, 3, 4, 0, 0, 2],
                               [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 2, 0]], dtype=int)
    edge_attr = torch.randn(12, 7)
    data = mydata.Data(x, edge_index, edge_attr)

    added = remove_self_loops(data)
    torch.testing.assert_close(added.x, x, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(added.edge_index, torch.tensor(
        [[0, 1, 1, 2, 3, 3, 4, 0, 0, 2],
         [1, 0, 2, 3, 2, 4, 0, 4, 2, 0]]
    ), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(added.edge_attr, torch.cat([edge_attr, torch.zeros(3, 7)])[
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 11]
    ], rtol=1e-9, atol=1e-9)


def test_utils_degree1():
    edge_index = torch.tensor([[0, 1, 1, 1, 2, 3, 3, 3, 4, 0, 0, 2],
                               [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 2, 0]], dtype=int)
    torch.testing.assert_close(degree(edge_index[0], 5), torch.tensor(
        [3, 3, 2, 3, 1]
    ), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(degree(edge_index[1], 5), torch.tensor(
        [3, 2, 3, 2, 2]
    ), rtol=1e-9, atol=1e-9)


def test_utils_degree2():
    edge_index = torch.tensor([[1, 2, 2, 3, 3, 4, 4, 5, 4, 1],
                               [2, 1, 3, 2, 4, 3, 5, 4, 1, 4]], dtype=int)
    torch.testing.assert_close(
        degree(edge_index[0], 7), degree(edge_index[1], 7), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(degree(edge_index[0], 7), torch.tensor([
        0, 2, 2, 2, 3, 1, 0
    ]), rtol=1e-9, atol=1e-9)
