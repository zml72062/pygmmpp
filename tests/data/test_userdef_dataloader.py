import sys
try:
    sys.path.append('.')
except:
    pass
import torch
from pygmmpp.data import Data, Batch, Dataset, DataLoader
from pygmmpp.data.collate import collate as lib_collate

def collate(cls, data_list):
    """
    A user-defined version of `collate`, which adds correct increments on
    `edge_index`-like tensor attributes.
    """
    batch = lib_collate(cls, data_list)
    
    if 'two_hop_edge_index' in batch.__dict__:
        batch_level, num_nodes = batch.batch_level, batch.num_nodes
        for i in range(len(batch.__dict__[f'ptr{batch_level}'])):
            slice_2 = batch.__dict__[f'two_hop_edge_index_slice{batch_level}']
            begin = slice_2[i]
            try:
                end = slice_2[i+1]
            except IndexError:
                end = batch.two_hop_edge_index.shape[1]
            batch.two_hop_edge_index[:, begin:end] = \
            batch.two_hop_edge_index[:, begin:end] + batch.__dict__[f'ptr{batch_level}'][i]
    return batch
    
class MyBatch:
    """
    A user-defined version of `Batch` which uses the collating function
    defined above.
    """
    def __init__(self, batch: Batch):
        self.batch = batch
    
    @staticmethod
    def from_data_list(cls, data_list) -> "MyBatch":
        # Please let `cls=Batch` here
        return MyBatch(collate(cls, data_list))

def test_userdef_batch_1():
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
    two_hop_neighbor = [
        torch.ones((2, 0), dtype=int),
        torch.tensor([[0, 2, 0, 4, 1, 3, 1, 5, 2, 4, 3, 5],
                      [2, 0, 4, 0, 3, 1, 5, 1, 4, 2, 3, 5]], dtype=int),
        torch.tensor([[0, 3, 1, 3, 1, 4, 2, 4],
                      [3, 0, 3, 1, 4, 1, 4, 2]], dtype=int),
        torch.tensor([[1, 3],
                      [3, 1]], dtype=int)
    ]

    data_list = [Data(x_i, edge_index_i, edge_attr_i, y_i) for
                 (x_i, edge_index_i, edge_attr_i, y_i)
                 in zip(x, edge_index, edge_attr, y)]
    # We do the same thing as batching `two_hop_neighbor` as an
    # `edge_index`-like tensor attribute, but use manually 
    # defined collation function to test against the library version
    for i in range(4):
        data_list[i].__set_tensor_attr__('two_hop_edge_index', two_hop_neighbor[i],
                                         collate_type='auto_collate', cat_dim=1, slicing=True)
    batch = MyBatch.from_data_list(Batch, data_list)

    data_list2 = [Data(x_i, edge_index_i, edge_attr_i, y_i) for
                (x_i, edge_index_i, edge_attr_i, y_i)
                in zip(x, edge_index, edge_attr, y)]
    for i in range(4):
        data_list2[i].__set_tensor_attr__('two_hop_edge_index', two_hop_neighbor[i],
                                         collate_type='edge_index', slicing=True)
    batch2 = Batch.from_data_list(data_list2)


    assert batch.batch.keys() == batch2.keys()
    for k in batch.batch:
        val1 = batch.batch.__dict__[k]
        val2 = batch2.__dict__[k]
        if isinstance(val1, torch.Tensor):
            torch.testing.assert_close(val1, val2)
        elif k not in {'edge_index_set', 'cat_dim_dict', 'inc_dict'}:
            assert val1 == val2

def test_userdef_batch_2():
    torch.manual_seed(124)
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
    two_hop_neighbor = [
        torch.ones((2, 0), dtype=int),
        torch.tensor([[0, 2, 0, 4, 1, 3, 1, 5, 2, 4, 3, 5],
                      [2, 0, 4, 0, 3, 1, 5, 1, 4, 2, 3, 5]], dtype=int),
        torch.tensor([[0, 3, 1, 3, 1, 4, 2, 4],
                      [3, 0, 3, 1, 4, 1, 4, 2]], dtype=int),
        torch.tensor([[1, 3],
                      [3, 1]], dtype=int)
    ]

    data_list = [Data(x_i, edge_index_i, edge_attr_i, y_i) for
                 (x_i, edge_index_i, edge_attr_i, y_i)
                 in zip(x, edge_index, edge_attr, y)]
    # We do the same thing as batching `two_hop_neighbor` as an
    # `edge_index`-like tensor attribute, but use manually 
    # defined collation function to test against the library version
    for i in range(4):
        data_list[i].__set_tensor_attr__('two_hop_edge_index', two_hop_neighbor[i],
                                         collate_type='auto_collate', cat_dim=1, slicing=True)
    batch01 = MyBatch.from_data_list(Batch, data_list[:2])
    batch02 = MyBatch.from_data_list(Batch, data_list[2:])
    batch = MyBatch.from_data_list(Batch, [batch01.batch, batch02.batch])

    data_list2 = [Data(x_i, edge_index_i, edge_attr_i, y_i) for
                (x_i, edge_index_i, edge_attr_i, y_i)
                in zip(x, edge_index, edge_attr, y)]
    for i in range(4):
        data_list2[i].__set_tensor_attr__('two_hop_edge_index', two_hop_neighbor[i],
                                         collate_type='edge_index', slicing=True)
    batch11 = Batch.from_data_list(data_list2[:2])
    batch12 = Batch.from_data_list(data_list2[2:])
    batch2 = Batch.from_data_list([batch11, batch12])

    assert batch.batch.keys() == batch2.keys()
    for k in batch.batch:
        val1 = batch.batch.__dict__[k]
        val2 = batch2.__dict__[k]
        if isinstance(val1, torch.Tensor):
            torch.testing.assert_close(val1, val2)
        elif k not in {'edge_index_set', 'cat_dim_dict', 'inc_dict'}:
            assert val1 == val2

def test_userdef_dataloader():
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
    two_hop_neighbor = [
        torch.ones((2, 0), dtype=int),
        torch.tensor([[0, 2, 0, 4, 1, 3, 1, 5, 2, 4, 3, 5],
                      [2, 0, 4, 0, 3, 1, 5, 1, 4, 2, 3, 5]], dtype=int),
        torch.tensor([[0, 3, 1, 3, 1, 4, 2, 4],
                      [3, 0, 3, 1, 4, 1, 4, 2]], dtype=int),
        torch.tensor([[1, 3],
                      [3, 1]], dtype=int)
    ]

    data_list = [Data(x_i, edge_index_i, edge_attr_i, y_i) for
                 (x_i, edge_index_i, edge_attr_i, y_i)
                 in zip(x, edge_index, edge_attr, y)]
    # We do the same thing as batching `two_hop_neighbor` as an
    # `edge_index`-like tensor attribute, but use manually 
    # defined collation function to test against the library version
    for i in range(4):
        data_list[i].__set_tensor_attr__('two_hop_edge_index', two_hop_neighbor[i],
                                         collate_type='auto_collate', cat_dim=1, slicing=True)
    dataset = Dataset()
    dataset.data_batch = Batch.from_data_list(data_list)
    dataset.indices = torch.arange(4)
    loader = DataLoader(dataset, 2, collator=collate)
    
    batch01 = MyBatch.from_data_list(Batch, data_list[:2])
    batch02 = MyBatch.from_data_list(Batch, data_list[2:])
    for (b, g) in zip(loader, [batch01, batch02]):
        assert b.keys() == g.batch.keys()
        for k in b:
            val1 = b.__dict__[k]
            val2 = g.batch.__dict__[k]
            if isinstance(val1, torch.Tensor):
                torch.testing.assert_close(val1, val2)
            else:
                assert val1 == val2

    
