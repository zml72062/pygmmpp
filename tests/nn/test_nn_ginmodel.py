import sys
try:
    sys.path.append('.')
except:
    pass
import torch
import torch.nn
from pygmmpp.nn.gin_conv import GINConv as myGINConv
from pygmmpp.nn.pool import GlobalPool
from torch_geometric.nn import global_add_pool
from pygmmpp.nn.model import Model, MLP
from torch_geometric.nn.models.basic_gnn import GIN
from typing import Optional, Dict, Union, Callable, Any
from pygmmpp.datasets.tu_dataset import TUDataset as myTU
from torch_geometric.datasets import TUDataset as pyTU
from pygmmpp.data.dataloader import DataLoader as myLoader
from torch_geometric.loader import DataLoader as pyLoader
from torch.optim import Adam
import torch.nn.functional as F


class GINModel(Model):
    def init_layer(self, in_channels: int,
                   out_channels: int,
                   mlp_hidden_channels: int,
                   mlp_dropout: float = 0.0,
                   mlp_norm: Optional[str] = None,
                   eps: float = 0.0,
                   train_eps: bool = False) -> torch.nn.Module:

        nn: torch.nn.Module = MLP(in_channels,
                                  mlp_hidden_channels,
                                  1,
                                  out_channels,
                                  mlp_dropout,
                                  norm=mlp_norm)
        return myGINConv(nn, eps, train_eps)

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 mlp_hidden_channels: int,
                 out_channels: Optional[int] = None,
                 dropout: float = 0,
                 residual: Optional[str] = None,
                 norm: Optional[str] = None,
                 relu_first: bool = False,
                 mlp_dropout: float = 0.0,
                 mlp_norm: Optional[str] = None,
                 eps: float = 0.0,
                 train_eps: bool = False):
        super().__init__(in_channels,
                         hidden_channels,
                         num_layers,
                         out_channels,
                         dropout,
                         residual,
                         norm,
                         relu_first,
                         mlp_hidden_channels=mlp_hidden_channels,
                         mlp_dropout=mlp_dropout,
                         mlp_norm=mlp_norm,
                         eps=eps,
                         train_eps=train_eps)


class pyGINModel(GIN):
    def __init__(self, in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 out_channels: Optional[int] = None,
                 dropout: float = 0,
                 act: Union[str, Callable, None] = "relu",
                 act_first: bool = False,
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 norm: Union[str, Callable, None] = None,
                 norm_kwargs: Optional[Dict[str, Any]] = None,
                 jk: Optional[str] = None,
                 **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers, out_channels,
                         dropout, act, act_first, act_kwargs, norm, norm_kwargs,
                         jk, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()


def test_nn_ginmodel_forward():
    torch.manual_seed(192)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int)
    torch.manual_seed(1992)
    mymodel = GINModel(10, 50, 4, 50,  residual='cat', norm='batch_norm', mlp_norm='batch_norm',
                       eps=0.1, train_eps=True)
    torch.manual_seed(1992)
    pymodel = pyGINModel(10, 50, 4, norm='BatchNorm', jk='cat',
                         eps=0.1, train_eps=True)

    assert len([p for p in mymodel.parameters()]) == len(
        [p for p in pymodel.parameters()])
    torch.testing.assert_close(mymodel(x, edge_index=edge_index),
                               pymodel(x=x, edge_index=edge_index),
                               rtol=1e-6, atol=1e-6)


def test_nn_ginmodel_train_epoch_tu():
    mytu = myTU('./data/tests/my', name='MUTAG')
    pytu = pyTU('./data/tests/py', name='MUTAG')

    myloader = myLoader(mytu, 32)
    pyloader = pyLoader(pytu, 32)

    torch.manual_seed(1992)
    mymodel = GINModel(7, 50, 4, 50, out_channels=2, residual='cat', norm='batch_norm', mlp_norm='batch_norm',
                       eps=0.1, train_eps=True)
    myoptim = Adam(mymodel.parameters(), lr=0.01)
    torch.manual_seed(1992)
    pymodel = pyGINModel(7, 50, 4, out_channels=2, norm='BatchNorm', jk='cat',
                         eps=0.1, train_eps=True)
    pyoptim = Adam(pymodel.parameters(), lr=0.01)

    mymodel.train()
    for batch in myloader:
        x, edge_index, y, batch = batch.x, batch.edge_index, batch.y, batch.batch0
        myoptim.zero_grad()
        myloss = F.nll_loss(F.log_softmax(
            GlobalPool()(mymodel(x, edge_index=edge_index),
                         batch), dim=1), y)
        myloss.backward()
        myoptim.step()

    pymodel.train()
    for batch in pyloader:
        x, edge_index, y, batch = batch.x, batch.edge_index, batch.y, batch.batch
        pyoptim.zero_grad()
        pyloss = F.nll_loss(F.log_softmax(
            global_add_pool(pymodel(x=x, edge_index=edge_index), batch),
            dim=1), y)
        pyloss.backward()
        pyoptim.step()

    torch.testing.assert_close(myloss, pyloss, rtol=1e-5, atol=1e-5)


def test_nn_ginmodel_eval_epoch_tu():
    mytu = myTU('./data/tests/my', name='MUTAG')
    pytu = pyTU('./data/tests/py', name='MUTAG')

    myloader = myLoader(mytu, 32)
    pyloader = pyLoader(pytu, 32)

    torch.manual_seed(1992)
    mymodel = GINModel(7, 50, 4, 50, out_channels=2, residual='cat', norm='batch_norm', mlp_norm='batch_norm',
                       eps=0.1, train_eps=True)
    torch.manual_seed(1992)
    pymodel = pyGINModel(7, 50, 4, out_channels=2, norm='BatchNorm', jk='cat',
                         eps=0.1, train_eps=True)

    mymodel.eval()
    for batch in myloader:
        x, edge_index, y, batch = batch.x, batch.edge_index, batch.y, batch.batch0
        myloss = F.nll_loss(F.log_softmax(
            GlobalPool()(mymodel(x, edge_index=edge_index),
                         batch), dim=1), y)

    pymodel.eval()
    for batch in pyloader:
        x, edge_index, y, batch = batch.x, batch.edge_index, batch.y, batch.batch
        pyloss = F.nll_loss(F.log_softmax(
            global_add_pool(pymodel(x=x, edge_index=edge_index), batch),
            dim=1), y)

    torch.testing.assert_close(myloss, pyloss, rtol=1e-5, atol=1e-5)

def test_nn_ginmodel_forward_layernorm():
    torch.manual_seed(192)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int)
    torch.manual_seed(1992)
    mymodel = GINModel(10, 50, 4, 50,  residual='cat', norm='layer_norm', mlp_norm='layer_norm',
                       eps=0.1, train_eps=True)
    torch.manual_seed(1992)
    pymodel = pyGINModel(10, 50, 4, norm='LayerNorm', norm_kwargs={'mode': 'node'}, jk='cat',
                         eps=0.1, train_eps=True)

    assert len([p for p in mymodel.parameters()]) == len(
        [p for p in pymodel.parameters()])
    torch.testing.assert_close(mymodel(x, edge_index=edge_index),
                               pymodel(x=x, edge_index=edge_index),
                               rtol=1e-6, atol=1e-6)


def test_nn_ginmodel_train_epoch_tu_layernorm():
    mytu = myTU('./data/tests/my', name='MUTAG')
    pytu = pyTU('./data/tests/py', name='MUTAG')

    myloader = myLoader(mytu, 32)
    pyloader = pyLoader(pytu, 32)

    torch.manual_seed(1992)
    mymodel = GINModel(7, 50, 4, 50, out_channels=2, residual='cat', norm='layer_norm', mlp_norm='layer_norm',
                       eps=0.1, train_eps=True)
    myoptim = Adam(mymodel.parameters(), lr=0.01)
    torch.manual_seed(1992)
    pymodel = pyGINModel(7, 50, 4, out_channels=2, norm='LayerNorm', norm_kwargs={'mode': 'node'}, jk='cat',
                         eps=0.1, train_eps=True)
    pyoptim = Adam(pymodel.parameters(), lr=0.01)

    mymodel.train()
    for batch in myloader:
        x, edge_index, y, batch = batch.x, batch.edge_index, batch.y, batch.batch0
        myoptim.zero_grad()
        myloss = F.nll_loss(F.log_softmax(
            GlobalPool()(mymodel(x, edge_index=edge_index),
                         batch), dim=1), y)
        myloss.backward()
        myoptim.step()

    pymodel.train()
    for batch in pyloader:
        x, edge_index, y, batch = batch.x, batch.edge_index, batch.y, batch.batch
        pyoptim.zero_grad()
        pyloss = F.nll_loss(F.log_softmax(
            global_add_pool(pymodel(x=x, edge_index=edge_index), batch),
            dim=1), y)
        pyloss.backward()
        pyoptim.step()

    torch.testing.assert_close(myloss, pyloss, rtol=1e-5, atol=1e-5)


def test_nn_ginmodel_eval_epoch_tu_layernorm():
    mytu = myTU('./data/tests/my', name='MUTAG')
    pytu = pyTU('./data/tests/py', name='MUTAG')

    myloader = myLoader(mytu, 32)
    pyloader = pyLoader(pytu, 32)

    torch.manual_seed(1992)
    mymodel = GINModel(7, 50, 4, 50, out_channels=2, residual='cat', norm='layer_norm', mlp_norm='layer_norm',
                       eps=0.1, train_eps=True)
    torch.manual_seed(1992)
    pymodel = pyGINModel(7, 50, 4, out_channels=2, norm='LayerNorm', norm_kwargs={'mode': 'node'}, jk='cat',
                         eps=0.1, train_eps=True)

    mymodel.eval()
    for batch in myloader:
        x, edge_index, y, batch = batch.x, batch.edge_index, batch.y, batch.batch0
        myloss = F.nll_loss(F.log_softmax(
            GlobalPool()(mymodel(x, edge_index=edge_index),
                         batch), dim=1), y)

    pymodel.eval()
    for batch in pyloader:
        x, edge_index, y, batch = batch.x, batch.edge_index, batch.y, batch.batch
        pyloss = F.nll_loss(F.log_softmax(
            global_add_pool(pymodel(x=x, edge_index=edge_index), batch),
            dim=1), y)

    torch.testing.assert_close(myloss, pyloss, rtol=1e-5, atol=1e-5)
