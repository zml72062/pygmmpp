import sys
try:
    sys.path.append('.')
except:
    pass
from pygmmpp.datasets.qm9 import QM9 as myQM9
from torch_geometric.datasets.qm9 import QM9 as pyQM9
from pygmmpp.data import DataLoader as myDataLoader
from torch_geometric.loader import DataLoader as pyDataLoader
from pygmmpp.data import Dataset as myDataset
from torch_geometric.data import Dataset as pyDataset
import torch


def test_dataset_qm9():
    myd = myQM9(root='./data/tests/QM9/my')
    pyd = pyQM9(root='./data/tests/QM9/py')

    assert len(myd) == len(pyd)
    assert myd.num_node_features == pyd.num_node_features
    assert myd.num_edge_features == pyd.num_edge_features
    for i in [32, 127, 28983, 102342]:
        for key in {'x', 'y', 'edge_index', 'edge_attr', 'pos', 'z', 'name', 'idx'}:
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
        for key in {'x', 'y', 'edge_index', 'edge_attr', 'pos', 'z', 'name', 'idx'}:
            val1 = myd[i].__dict__[key]
            val2 = pyd[i][key]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2


def test_dataset_qm9_slicing():
    myd = myQM9(root='./data/tests/QM9/my')
    pyd = pyQM9(root='./data/tests/QM9/py')

    sp1, sp2 = int(0.8*len(myd)), int(0.9*len(myd))

    myd_test: myDataset = myd[sp1:sp2]
    pyd_test: pyDataset = pyd[sp1:sp2]

    assert len(myd_test) == len(pyd_test)
    assert len(myd_test) == sp2 - sp1

    for (data1, data2) in zip(myd_test, pyd_test):
        for key in {'x', 'y', 'edge_index', 'edge_attr', 'pos', 'z', 'name', 'idx'}:
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
        for key in {'x', 'y', 'edge_index', 'edge_attr', 'pos', 'z', 'name', 'idx'}:
            val1 = data1.__dict__[key]
            val2 = data2[key]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2


def test_dataloader_load_qm9():
    myd = myQM9(root='./data/tests/QM9/my')
    pyd = pyQM9(root='./data/tests/QM9/py')
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
        for key in ['x', 'y', 'edge_index', 'edge_attr', 'name']:
            val1 = mybatch.__dict__[key]
            val2 = pybatch[key]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2

    # Due to the different collating methods between `pygmmpp` and `torch_geometric`,
    # we treat features ['pos', 'z', 'idx'] differently
    for (mybatch, pybatch) in zip(my_test_loader, py_test_loader):
        for key in ['pos', 'z', 'idx']:
            if key == 'idx':
                val1 = torch.tensor(mybatch.__dict__[key])
            else:
                val1 = torch.cat(mybatch.__dict__[key])
            val2 = pybatch[key]
            torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)

    for (mybatch, pybatch) in zip(my_train_loader, py_train_loader):
        for key in ['x', 'y', 'edge_index', 'edge_attr', 'name']:
            val1 = mybatch.__dict__[key]
            val2 = pybatch[key]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
            else:
                assert val1 == val2

    for (mybatch, pybatch) in zip(my_train_loader, py_train_loader):
        for key in ['pos', 'z', 'idx']:
            if key == 'idx':
                val1 = torch.tensor(mybatch.__dict__[key])
            else:
                val1 = torch.cat(mybatch.__dict__[key])
            val2 = pybatch[key]
            torch.testing.assert_close(val1, val2, rtol=1e-9, atol=1e-9)
