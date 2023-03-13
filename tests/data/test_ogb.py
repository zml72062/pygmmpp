import sys
try:
    sys.path.append('.')
except:
    pass
from pygmmpp.datasets.ogb_graph import OGBG as myOGB
from ogb.graphproppred import PygGraphPropPredDataset as pyOGB
from pygmmpp.data import DataLoader as myDataLoader
from torch_geometric.loader import DataLoader as pyDataLoader
from pygmmpp.data import Dataset as myDataset
from torch_geometric.data import Dataset as pyDataset
import torch


def test_dataset_ogbg_molhiv():
    myd = myOGB(root='./data/tests/OGB/my', name='ogbg-molhiv')
    pyd = pyOGB(root='./data/tests/OGB/py', name='ogbg-molhiv')

    assert len(myd) == len(pyd)
    assert myd.num_node_features == pyd.num_node_features
    assert myd.num_edge_features == pyd.num_edge_features
    for i in [32, 127, 28983, 10242]:
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
    for i in [32, 127, 28983, 10342]:
        for key in {'x', 'y', 'edge_index', 'edge_attr'}:
            val1 = myd[i].__dict__[key]
            val2 = pyd[i][key]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2


def test_dataset_ogbg_molpcba_slicing():
    myd = myOGB(root='./data/tests/OGB/my', name='ogbg-molpcba')
    pyd = pyOGB(root='./data/tests/OGB/py', name='ogbg-molpcba')

    sp1, sp2 = int(0.8*len(myd)), int(0.9*len(myd))

    myd_test: myDataset = myd[sp1:sp2]
    pyd_test: pyDataset = pyd[sp1:sp2]

    assert len(myd_test) == len(pyd_test)
    assert len(myd_test) == sp2 - sp1

    for (data1, data2) in zip(myd_test, pyd_test):
        for key in {'x', 'y', 'edge_index', 'edge_attr'}:
            val1 = data1.__dict__[key]
            val2 = data2[key]
            if key == 'y':
                val1 = torch.nan_to_num(val1, nan=1882937)
                val2 = torch.nan_to_num(val2, nan=1882937)
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2
    
    for label in ['valid', 'test']:
        myd_test = myd[myd.get_idx_split()[label]]
        pyd_test = pyd[pyd.get_idx_split()[label]]

        assert len(myd_test) == len(pyd_test)

        for (data1, data2) in zip(myd_test, pyd_test):
            for key in {'x', 'y', 'edge_index', 'edge_attr'}:
                val1 = data1.__dict__[key]
                val2 = data2[key]
                if key == 'y':
                    val1 = torch.nan_to_num(val1, nan=1882937)
                    val2 = torch.nan_to_num(val2, nan=1882937)
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
            if key == 'y':
                val1 = torch.nan_to_num(val1, nan=1882937)
                val2 = torch.nan_to_num(val2, nan=1882937)
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2
    
    for label in ['valid', 'test']:
        myd_test = myd[myd.get_idx_split()[label]]
        pyd_test = pyd[pyd.get_idx_split()[label]]

        assert len(myd_test) == len(pyd_test)

        for (data1, data2) in zip(myd_test, pyd_test):
            for key in {'x', 'y', 'edge_index', 'edge_attr'}:
                val1 = data1.__dict__[key]
                val2 = data2[key]
                if key == 'y':
                    val1 = torch.nan_to_num(val1, nan=1882937)
                    val2 = torch.nan_to_num(val2, nan=1882937)
                if isinstance(val1, torch.Tensor):
                    torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
                else:
                    assert val1 == val2
    


def test_dataloader_load_ogbg_molhiv():
    myd = myOGB(root='./data/tests/OGB/my', name='ogbg-molhiv')
    pyd = pyOGB(root='./data/tests/OGB/py', name='ogbg-molhiv')
    
    my_train_idx = myd.get_idx_split()['train']
    py_train_idx = pyd.get_idx_split()['train']
    my_valid_idx = myd.get_idx_split()['valid']
    py_valid_idx = pyd.get_idx_split()['valid']
    my_test_idx = myd.get_idx_split()['test']
    py_test_idx = pyd.get_idx_split()['test']

    # In order to produce the same batches after shuffling,
    # we specify the random seed and `generator` argument in `DataLoader`
    mygenerator = torch.Generator()
    mygenerator.manual_seed(1002)
    my_train_loader = myDataLoader(
        myd[my_train_idx], batch_size=32, shuffle=True, generator=mygenerator
    )
    my_valid_loader = myDataLoader(myd[my_valid_idx], batch_size=32, shuffle=False)
    my_test_loader = myDataLoader(myd[my_test_idx], batch_size=32, shuffle=False)

    pygenerator = torch.Generator()
    pygenerator.manual_seed(1002)
    py_train_loader = pyDataLoader(
        pyd[py_train_idx], batch_size=32, shuffle=True, generator=pygenerator
    )
    py_valid_loader = pyDataLoader(pyd[py_valid_idx], batch_size=32, shuffle=False)
    py_test_loader = pyDataLoader(pyd[py_test_idx], batch_size=32, shuffle=False)

    for (mybatch, pybatch) in zip(my_test_loader, py_test_loader):
        for key in ['x', 'y', 'edge_index', 'edge_attr']:
            val1 = mybatch.__dict__[key]
            val2 = pybatch[key]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2

    for (mybatch, pybatch) in zip(my_valid_loader, py_valid_loader):
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
