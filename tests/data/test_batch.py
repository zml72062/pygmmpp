import sys
try:
    sys.path.append('.')
except:
    pass
from pygmmpp.data import Data, Batch
import torch
import numpy
from typing import List


def test_batch_collate_1():
    torch.manual_seed(42)
    x = [
        torch.randn(6, 10),
        torch.randn(6, 10),
        torch.randn(5, 10),
        torch.randn(4, 10)
    ]
    edge_index = [
        torch.tensor([[0, 1, 0, 2, 1, 2, 3, 4, 3, 5, 4, 5],
                      [1, 0, 2, 0, 2, 1, 4, 3, 5, 3, 5, 4]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                      [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 2],
                      [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 2, 0]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 0, 2],
                      [1, 0, 2, 1, 3, 2, 0, 3, 2, 0]], dtype=int)
    ]
    edge_attr = [
        torch.randn(12, 7),
        torch.randn(14, 7),
        torch.randn(12, 7),
        torch.randn(10, 7)
    ]
    y = [
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3)
    ]
    data_list = [Data(x_i, edge_index_i, edge_attr_i, y_i) for
                 (x_i, edge_index_i, edge_attr_i, y_i)
                 in zip(x, edge_index, edge_attr, y)]
    batch: Batch = Batch.from_data_list(data_list)
    assert set(batch.keys()) == {'x', 'edge_index', 'edge_attr', 'y',
                                 'inc_dict', 'cat_dim_dict', 'batch0',
                                 'ptr0', 'batch_level', 'edge_slice0'}
    assert batch.num_nodes == 21
    assert batch.num_edges == 48
    assert batch.num_node_features == 10
    assert batch.num_edge_features == 7
    assert batch.num_graphs == 4
    assert batch.batch_level == 0

    torch.testing.assert_close(batch.x, torch.cat(x), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(batch.edge_attr, torch.cat(edge_attr),
                               rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(batch.y, torch.cat(y), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(batch.edge_index, torch.cat(
        [edge_index[0], edge_index[1]+6, edge_index[2]+12, edge_index[3]+17],
        dim=1), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(batch.batch0, torch.tensor(
        [0]*6+[1]*6+[2]*5+[3]*4, dtype=int), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(batch.ptr0, torch.tensor(
        [0, 6, 12, 17]), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(batch.edge_slice0, torch.tensor(
        [0, 12, 26, 38]), rtol=1e-9, atol=1e-9)


def test_batch_separate_1():
    torch.manual_seed(172)
    x = [
        torch.randn(6, 10),
        torch.randn(6, 10),
        torch.randn(5, 10),
        torch.randn(4, 10)
    ]
    edge_index = [
        torch.tensor([[0, 1, 0, 2, 1, 2, 3, 4, 3, 5, 4, 5],
                      [1, 0, 2, 0, 2, 1, 4, 3, 5, 3, 5, 4]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                      [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 2],
                      [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 2, 0]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 0, 2],
                      [1, 0, 2, 1, 3, 2, 0, 3, 2, 0]], dtype=int)
    ]
    edge_attr = [
        torch.randn(12, 7),
        torch.randn(14, 7),
        torch.randn(12, 7),
        torch.randn(10, 7)
    ]
    y = [
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3)
    ]
    data_list = [Data(x_i, edge_index_i, edge_attr_i, y_i) for
                 (x_i, edge_index_i, edge_attr_i, y_i)
                 in zip(x, edge_index, edge_attr, y)]
    batch: Batch = Batch.from_data_list(data_list)
    for i in range(4):
        data1 = batch[i]
        data2 = data_list[i]
        # test `data1 == data2`
        assert data1.keys() == data2.keys()
        for k in data1:
            val1 = data1.__dict__[k]
            val2 = data2.__dict__[k]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2


def test_batch_collate_2():
    torch.manual_seed(132)
    x = [
        torch.randn(6, 10),
        torch.randn(6, 10),
        torch.randn(5, 10),
        torch.randn(4, 10)
    ]
    edge_index = [
        torch.tensor([[0, 1, 0, 2, 1, 2, 3, 4, 3, 5, 4, 5],
                      [1, 0, 2, 0, 2, 1, 4, 3, 5, 3, 5, 4]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                      [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 2],
                      [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 2, 0]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 0, 2],
                      [1, 0, 2, 1, 3, 2, 0, 3, 2, 0]], dtype=int)
    ]
    edge_attr = [
        torch.randn(12, 7),
        torch.randn(14, 7),
        torch.randn(12, 7),
        torch.randn(10, 7)
    ]
    y = [
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3)
    ]
    data_list = [Data(x_i, edge_index_i, edge_attr_i, y_i) for
                 (x_i, edge_index_i, edge_attr_i, y_i)
                 in zip(x, edge_index, edge_attr, y)]
    g1 = Batch.from_data_list([data_list[0], data_list[1]])
    g2 = Batch.from_data_list([data_list[2], data_list[3]])
    batch: Batch = Batch.from_data_list([g1, g2])
    assert set(batch.keys()) == {'x', 'edge_index', 'edge_attr', 'y',
                                 'inc_dict', 'cat_dim_dict', 'batch0',
                                 'ptr0', 'batch1', 'ptr1', 'batch_level',
                                 'edge_slice0', 'edge_slice1'}
    assert batch.num_nodes == 21
    assert batch.num_edges == 48
    assert batch.num_node_features == 10
    assert batch.num_edge_features == 7
    assert batch.num_graphs == 2
    assert batch.batch_level == 1

    torch.testing.assert_close(batch.x, torch.cat(x), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(batch.edge_attr, torch.cat(edge_attr),
                               rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(batch.y, torch.cat(y), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(batch.edge_index, torch.cat(
        [edge_index[0], edge_index[1]+6, edge_index[2]+12, edge_index[3]+17],
        dim=1), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(batch.batch0, torch.tensor(
        [0]*6+[1]*6+[2]*5+[3]*4, dtype=int), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(batch.ptr0, torch.tensor(
        [0, 6, 12, 17]), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(batch.batch1, torch.tensor(
        [0]*12+[1]*9, dtype=int), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        batch.ptr1, torch.tensor([0, 12]), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(batch.edge_slice0, torch.tensor(
        [0, 12, 26, 38]), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        batch.edge_slice1, torch.tensor([0, 26]), rtol=1e-9, atol=1e-9)


def test_batch_separate_2():
    torch.manual_seed(2)
    x = [
        torch.randn(6, 10),
        torch.randn(6, 10),
        torch.randn(5, 10),
        torch.randn(4, 10)
    ]
    edge_index = [
        torch.tensor([[0, 1, 0, 2, 1, 2, 3, 4, 3, 5, 4, 5],
                      [1, 0, 2, 0, 2, 1, 4, 3, 5, 3, 5, 4]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                      [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 2],
                      [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 2, 0]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 0, 2],
                      [1, 0, 2, 1, 3, 2, 0, 3, 2, 0]], dtype=int)
    ]
    edge_attr = [
        torch.randn(12, 7),
        torch.randn(14, 7),
        torch.randn(12, 7),
        torch.randn(10, 7)
    ]
    y = [
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3)
    ]
    data_list = [Data(x_i, edge_index_i, edge_attr_i, y_i) for
                 (x_i, edge_index_i, edge_attr_i, y_i)
                 in zip(x, edge_index, edge_attr, y)]
    g1: Batch = Batch.from_data_list([data_list[0], data_list[1]])
    g2: Batch = Batch.from_data_list([data_list[2], data_list[3]])
    batch: Batch = Batch.from_data_list([g1, g2])
    b1, b2 = batch[0], batch[1]

    assert b1.keys() == g1.keys()
    for k in b1:
        val1 = b1.__dict__[k]
        val2 = g1.__dict__[k]
        if isinstance(val1, torch.Tensor):
            torch.testing.assert_close(val1, val2)
        else:
            assert val1 == val2

    assert b2.keys() == g2.keys()
    for k in b2:
        val1 = b2.__dict__[k]
        val2 = g2.__dict__[k]
        if isinstance(val1, torch.Tensor):
            torch.testing.assert_close(val1, val2)
        else:
            assert val1 == val2


def test_batch_slicing():
    torch.manual_seed(62)
    x = [
        torch.randn(6, 10),
        torch.randn(6, 10),
        torch.randn(5, 10),
        torch.randn(4, 10)
    ]
    edge_index = [
        torch.tensor([[0, 1, 0, 2, 1, 2, 3, 4, 3, 5, 4, 5],
                      [1, 0, 2, 0, 2, 1, 4, 3, 5, 3, 5, 4]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                      [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 2],
                      [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 2, 0]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 0, 2],
                      [1, 0, 2, 1, 3, 2, 0, 3, 2, 0]], dtype=int)
    ]
    edge_attr = [
        torch.randn(12, 7),
        torch.randn(14, 7),
        torch.randn(12, 7),
        torch.randn(10, 7)
    ]
    y = [
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3)
    ]
    data_list = [Data(x_i, edge_index_i, edge_attr_i, y_i) for
                 (x_i, edge_index_i, edge_attr_i, y_i)
                 in zip(x, edge_index, edge_attr, y)]
    batch: Batch = Batch.from_data_list(data_list)
    b1 = batch[1:]
    b2 = batch[:2]
    b3 = batch[[0, 3]]
    b4 = batch[numpy.array([2, 0], dtype=int)]
    b5 = batch[torch.tensor([0, 2, 3], dtype=int)]

    g1 = Batch.from_data_list(data_list[1:])
    g2 = Batch.from_data_list(data_list[:2])
    g3 = Batch.from_data_list([data_list[0], data_list[3]])
    g4 = Batch.from_data_list([data_list[2], data_list[0]])
    g5 = Batch.from_data_list([data_list[0], data_list[2], data_list[3]])

    assert b1.keys() == g1.keys()
    for k in b1:
        val1 = b1.__dict__[k]
        val2 = g1.__dict__[k]
        if isinstance(val1, torch.Tensor):
            torch.testing.assert_close(val1, val2)
        else:
            assert val1 == val2

    assert b2.keys() == g2.keys()
    for k in b2:
        val1 = b2.__dict__[k]
        val2 = g2.__dict__[k]
        if isinstance(val1, torch.Tensor):
            torch.testing.assert_close(val1, val2)
        else:
            assert val1 == val2

    assert b3.keys() == g3.keys()
    for k in b3:
        val1 = b3.__dict__[k]
        val2 = g3.__dict__[k]
        if isinstance(val1, torch.Tensor):
            torch.testing.assert_close(val1, val2)
        else:
            assert val1 == val2

    assert b4.keys() == g4.keys()
    for k in b4:
        val1 = b4.__dict__[k]
        val2 = g4.__dict__[k]
        if isinstance(val1, torch.Tensor):
            torch.testing.assert_close(val1, val2)
        else:
            assert val1 == val2

    assert b5.keys() == g5.keys()
    for k in b5:
        val1 = b5.__dict__[k]
        val2 = g5.__dict__[k]
        if isinstance(val1, torch.Tensor):
            torch.testing.assert_close(val1, val2)
        else:
            assert val1 == val2


def test_batch_rebatch_1():
    torch.manual_seed(1992)
    x = [
        torch.randn(6, 10),
        torch.randn(6, 10),
        torch.randn(5, 10),
        torch.randn(4, 10)
    ]
    edge_index = [
        torch.tensor([[0, 1, 0, 2, 1, 2, 3, 4, 3, 5, 4, 5],
                      [1, 0, 2, 0, 2, 1, 4, 3, 5, 3, 5, 4]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                      [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 2],
                      [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 2, 0]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 0, 2],
                      [1, 0, 2, 1, 3, 2, 0, 3, 2, 0]], dtype=int)
    ]
    edge_attr = [
        torch.randn(12, 7),
        torch.randn(14, 7),
        torch.randn(12, 7),
        torch.randn(10, 7)
    ]
    y = [
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3)
    ]
    data_list = [Data(x_i, edge_index_i, edge_attr_i, y_i) for
                 (x_i, edge_index_i, edge_attr_i, y_i)
                 in zip(x, edge_index, edge_attr, y)]
    batch: Batch = Batch.from_data_list(data_list)

    def to_data_list(batch: Batch) -> List[Data]:
        return [batch[i] for i in range(batch.num_graphs)]

    batch_again: Batch = Batch.from_data_list(to_data_list(batch))

    assert batch.keys() == batch_again.keys()
    for k in batch:
        val1 = batch.__dict__[k]
        val2 = batch_again.__dict__[k]
        if isinstance(val1, torch.Tensor):
            torch.testing.assert_close(val1, val2)
        else:
            assert val1 == val2


def test_batch_rebatch_2():
    torch.manual_seed(2002)
    x = [
        torch.randn(6, 10),
        torch.randn(6, 10),
        torch.randn(5, 10),
        torch.randn(4, 10)
    ]
    edge_index = [
        torch.tensor([[0, 1, 0, 2, 1, 2, 3, 4, 3, 5, 4, 5],
                      [1, 0, 2, 0, 2, 1, 4, 3, 5, 3, 5, 4]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                      [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 2],
                      [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 2, 0]], dtype=int),
        torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 0, 2],
                      [1, 0, 2, 1, 3, 2, 0, 3, 2, 0]], dtype=int)
    ]
    edge_attr = [
        torch.randn(12, 7),
        torch.randn(14, 7),
        torch.randn(12, 7),
        torch.randn(10, 7)
    ]
    y = [
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3)
    ]
    data_list = [Data(x_i, edge_index_i, edge_attr_i, y_i) for
                 (x_i, edge_index_i, edge_attr_i, y_i)
                 in zip(x, edge_index, edge_attr, y)]
    g1 = Batch.from_data_list([data_list[0], data_list[1]])
    g2 = Batch.from_data_list([data_list[2], data_list[3]])
    batch: Batch = Batch.from_data_list([g1, g2])

    def to_data_list(batch: Batch) -> List[Data]:
        return [batch[i] for i in range(batch.num_graphs)]

    l = to_data_list(batch)

    assert l[0].keys() == g1.keys()
    for k in l[0]:
        val1 = l[0].__dict__[k]
        val2 = g1.__dict__[k]
        if isinstance(val1, torch.Tensor):
            torch.testing.assert_close(val1, val2)
        else:
            assert val1 == val2

    assert l[1].keys() == g2.keys()
    for k in l[1]:
        val1 = l[1].__dict__[k]
        val2 = g2.__dict__[k]
        if isinstance(val1, torch.Tensor):
            torch.testing.assert_close(val1, val2)
        else:
            assert val1 == val2
