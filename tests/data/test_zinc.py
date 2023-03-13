import sys
try:
    sys.path.append('.')
except:
    pass
from pygmmpp.datasets.zinc import ZINC as myZINC
from torch_geometric.datasets.zinc import ZINC as pyZINC
from pygmmpp.data import DataLoader as myDataLoader
from torch_geometric.loader import DataLoader as pyDataLoader
from pygmmpp.data import Dataset as myDataset
from torch_geometric.data import Dataset as pyDataset
import torch


def test_dataset_zinc():
    myd = myZINC(root='./data/tests/ZINC/my', subset=True, split='val')
    pyd = pyZINC(root='./data/tests/ZINC/py', subset=True, split='val')

    assert len(myd) == len(pyd)
    assert myd.num_node_features == pyd.num_node_features
    assert myd.num_edge_features == pyd.num_edge_features
    for i in [32, 127, 738, 997]:
        for key in {'x', 'y', 'edge_index', 'edge_attr'}:
            val1 = myd[i].__dict__[key]
            val2 = pyd[i][key]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2

    torch.manual_seed(232)
    myd = myd.shuffle()
    torch.manual_seed(232)
    pyd = pyd.shuffle()
    for i in [32, 127, 738, 997]:
        for key in {'x', 'y', 'edge_index', 'edge_attr'}:
            val1 = myd[i].__dict__[key]
            val2 = pyd[i][key]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2

def test_dataset_zinc_full():
    myd = myZINC(root='./data/tests/ZINC-full/my', split='train')
    pyd = pyZINC(root='./data/tests/ZINC-full/py', split='train')

    assert len(myd) == len(pyd)
    assert myd.num_node_features == pyd.num_node_features
    assert myd.num_edge_features == pyd.num_edge_features
    for i in [32, 127, 28983, 102342]:
        for key in {'x', 'y', 'edge_index', 'edge_attr'}:
            val1 = myd[i].__dict__[key]
            val2 = pyd[i][key]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2

    torch.manual_seed(32)
    myd = myd.shuffle()
    torch.manual_seed(32)
    pyd = pyd.shuffle()
    for i in [32, 127, 28983, 102342]:
        for key in {'x', 'y', 'edge_index', 'edge_attr'}:
            val1 = myd[i].__dict__[key]
            val2 = pyd[i][key]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2


def test_dataset_zinc_slicing():
    myd = myZINC(root='./data/tests/ZINC-full/my', split='val')
    pyd = pyZINC(root='./data/tests/ZINC-full/py', split='val')

    sp1, sp2 = int(0.8*len(myd)), int(0.9*len(myd))

    myd_test: myDataset = myd[sp1:sp2]
    pyd_test: pyDataset = pyd[sp1:sp2]

    assert len(myd_test) == len(pyd_test)
    assert len(myd_test) == sp2 - sp1

    for (data1, data2) in zip(myd_test, pyd_test):
        for key in {'x', 'y', 'edge_index', 'edge_attr'}:
            val1 = data1.__dict__[key]
            val2 = data2[key]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2

    torch.manual_seed(1912)
    myd_test = myd_test.shuffle()
    torch.manual_seed(1912)
    pyd_test = pyd_test.shuffle()

    assert len(myd_test) == len(pyd_test)
    assert len(myd_test) == sp2 - sp1

    for (data1, data2) in zip(myd_test, pyd_test):
        for key in {'x', 'y', 'edge_index', 'edge_attr'}:
            val1 = data1.__dict__[key]
            val2 = data2[key]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2


def test_dataloader_load_zinc():
    myd = myZINC(root='./data/tests/ZINC/my', subset=True, split='train')
    pyd = pyZINC(root='./data/tests/ZINC/py', subset=True, split='train')
    sp = int(0.8*len(myd))

    # In order to produce the same batches after shuffling,
    # we specify the random seed and `generator` argument in `DataLoader`
    mygenerator = torch.Generator()
    mygenerator.manual_seed(1002)
    my_train_loader = myDataLoader(
        myd[:sp], batch_size=32, shuffle=True, generator=mygenerator
    )
    my_test_loader = myDataLoader(myd[sp:], batch_size=32, shuffle=False)

    pygenerator = torch.Generator()
    pygenerator.manual_seed(1002)
    py_train_loader = pyDataLoader(
        pyd[:sp], batch_size=32, shuffle=True, generator=pygenerator
    )
    py_test_loader = pyDataLoader(pyd[sp:], batch_size=32, shuffle=False)

    for (mybatch, pybatch) in zip(my_test_loader, py_test_loader):
        for key in ['x', 'y', 'edge_index', 'edge_attr']:
            val1 = mybatch.__dict__[key]
            val2 = pybatch[key]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2

    for (mybatch, pybatch) in zip(my_train_loader, py_train_loader):
        for key in ['x', 'y', 'edge_index', 'edge_attr']:
            val1 = mybatch.__dict__[key]
            val2 = pybatch[key]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2
