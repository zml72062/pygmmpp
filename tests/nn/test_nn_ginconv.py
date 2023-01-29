import sys
try:
    sys.path.append('.')
except:
    pass
import torch
import torch.nn
from pygmmpp.nn.gin_conv import GINConv as myGINConv
from pygmmpp.nn.gin_conv import GINEConv as myGINEConv
from torch_geometric.nn import GINConv as pyGINConv
from torch_geometric.nn import GINEConv as pyGINEConv
from pygmmpp.nn.pool import GlobalPool
from torch_geometric.nn import global_add_pool


def test_nn_ginconv_forward():
    torch.manual_seed(192)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int)

    nn = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 20)
    )
    torch.manual_seed(1292)
    myconv = myGINConv(nn, eps=0.1)
    torch.manual_seed(1292)
    pyconv = pyGINConv(nn, eps=0.1)

    torch.testing.assert_close(myconv(x, edge_index), pyconv(x, edge_index),
                               atol=1e-6, rtol=1e-6)


def test_nn_gineconv_forward():
    torch.manual_seed(1882)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int)
    edge_attr = torch.randn(14, 20)

    nn = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 20)
    )
    torch.manual_seed(12)
    myconv = myGINEConv(nn, eps=0.1, edge_dim=20, node_dim=10)
    torch.manual_seed(12)
    pyconv = pyGINEConv(nn, eps=0.1, edge_dim=20)

    torch.testing.assert_close(myconv(x, edge_index, edge_attr=edge_attr),
                               pyconv(x, edge_index, edge_attr=edge_attr),
                               atol=1e-6, rtol=1e-6)


def test_nn_ginconv_backward1():
    torch.manual_seed(182)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int)

    my_lin1 = torch.nn.Linear(10, 50)
    my_lin2 = torch.nn.Linear(50, 20)
    my_nn = torch.nn.Sequential(
        my_lin1,
        torch.nn.ReLU(),
        my_lin2
    )

    torch.manual_seed(122)
    myconv = myGINConv(my_nn, train_eps=True)
    my_out_lin = torch.nn.Linear(20, 1)

    py_lin1 = torch.nn.Linear(10, 50)
    py_lin2 = torch.nn.Linear(50, 20)
    py_nn = torch.nn.Sequential(
        py_lin1,
        torch.nn.ReLU(),
        py_lin2
    )

    torch.manual_seed(122)
    pyconv = pyGINConv(py_nn, train_eps=True)
    py_out_lin = torch.nn.Linear(20, 1)

    my_out = myconv(x, edge_index)
    py_out = pyconv(x, edge_index)
    my_y: torch.Tensor = GlobalPool(reduce='sum')(my_out_lin(my_out))
    py_y: torch.Tensor = global_add_pool(
        py_out_lin(py_out), batch=torch.zeros(6, dtype=int))
    my_y.backward()
    py_y.backward()

    torch.testing.assert_close(my_out_lin.weight.grad, py_out_lin.weight.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_out_lin.bias.grad, py_out_lin.bias.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin1.weight.grad, py_lin1.weight.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin1.bias.grad, py_lin1.bias.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin2.weight.grad, py_lin2.weight.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin2.bias.grad, py_lin2.bias.grad,
                               rtol=1e-6, atol=1e-6)
    assert myconv.eps.grad is not None
    torch.testing.assert_close(myconv.eps.grad, pyconv.eps.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(myconv.eps.grad, pyconv.eps.grad,
                               rtol=1e-6, atol=1e-6)


def test_nn_gineconv_backward1():
    torch.manual_seed(112)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int)
    edge_attr = torch.randn(14, 20)

    my_lin1 = torch.nn.Linear(10, 50)
    my_lin2 = torch.nn.Linear(50, 20)
    my_nn = torch.nn.Sequential(
        my_lin1,
        torch.nn.ReLU(),
        my_lin2
    )

    torch.manual_seed(122)
    myconv = myGINEConv(my_nn, train_eps=True, edge_dim=20, node_dim=10)
    my_out_lin = torch.nn.Linear(20, 1)

    py_lin1 = torch.nn.Linear(10, 50)
    py_lin2 = torch.nn.Linear(50, 20)
    py_nn = torch.nn.Sequential(
        py_lin1,
        torch.nn.ReLU(),
        py_lin2
    )

    torch.manual_seed(122)
    pyconv = pyGINEConv(py_nn, train_eps=True, edge_dim=20)
    py_out_lin = torch.nn.Linear(20, 1)

    my_out = myconv(x, edge_index, edge_attr=edge_attr)
    py_out = pyconv(x, edge_index, edge_attr)
    my_y: torch.Tensor = GlobalPool(reduce='sum')(my_out_lin(my_out))
    py_y: torch.Tensor = global_add_pool(
        py_out_lin(py_out), batch=torch.zeros(6, dtype=int))
    my_y.backward()
    py_y.backward()

    torch.testing.assert_close(my_out_lin.weight.grad, py_out_lin.weight.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_out_lin.bias.grad, py_out_lin.bias.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin1.weight.grad, py_lin1.weight.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin1.bias.grad, py_lin1.bias.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin2.weight.grad, py_lin2.weight.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin2.bias.grad, py_lin2.bias.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(myconv.lin.weight.grad, pyconv.lin.weight.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(myconv.lin.bias.grad, pyconv.lin.bias.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(myconv.eps.grad, pyconv.eps.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(myconv.eps.grad, pyconv.eps.grad,
                               rtol=1e-6, atol=1e-6)


def test_nn_ginconv_backward2():
    torch.manual_seed(182)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int)

    my_lin1 = torch.nn.Linear(10, 50)
    my_lin2 = torch.nn.Linear(50, 20)
    my_nn = torch.nn.Sequential(
        my_lin1,
        torch.nn.ReLU(),
        my_lin2
    )

    torch.manual_seed(122)
    myconv = myGINConv(my_nn)
    my_out_lin = torch.nn.Linear(20, 1)

    py_lin1 = torch.nn.Linear(10, 50)
    py_lin2 = torch.nn.Linear(50, 20)
    py_nn = torch.nn.Sequential(
        py_lin1,
        torch.nn.ReLU(),
        py_lin2
    )

    torch.manual_seed(122)
    pyconv = pyGINConv(py_nn)
    py_out_lin = torch.nn.Linear(20, 1)

    my_out = myconv(x, edge_index)
    py_out = pyconv(x, edge_index)
    my_y: torch.Tensor = GlobalPool(reduce='sum')(my_out_lin(my_out))
    py_y: torch.Tensor = global_add_pool(
        py_out_lin(py_out), batch=torch.zeros(6, dtype=int))
    my_y.backward()
    py_y.backward()

    torch.testing.assert_close(my_out_lin.weight.grad, py_out_lin.weight.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_out_lin.bias.grad, py_out_lin.bias.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin1.weight.grad, py_lin1.weight.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin1.bias.grad, py_lin1.bias.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin2.weight.grad, py_lin2.weight.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin2.bias.grad, py_lin2.bias.grad,
                               rtol=1e-6, atol=1e-6)
    assert not hasattr(myconv.eps, 'grad')


def test_nn_gineconv_backward2():
    torch.manual_seed(112)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int)
    edge_attr = torch.randn(14, 20)

    my_lin1 = torch.nn.Linear(10, 50)
    my_lin2 = torch.nn.Linear(50, 20)
    my_nn = torch.nn.Sequential(
        my_lin1,
        torch.nn.ReLU(),
        my_lin2
    )

    torch.manual_seed(122)
    myconv = myGINEConv(my_nn, edge_dim=20, node_dim=10)
    my_out_lin = torch.nn.Linear(20, 1)

    py_lin1 = torch.nn.Linear(10, 50)
    py_lin2 = torch.nn.Linear(50, 20)
    py_nn = torch.nn.Sequential(
        py_lin1,
        torch.nn.ReLU(),
        py_lin2
    )

    torch.manual_seed(122)
    pyconv = pyGINEConv(py_nn, edge_dim=20)
    py_out_lin = torch.nn.Linear(20, 1)

    my_out = myconv(x, edge_index, edge_attr=edge_attr)
    py_out = pyconv(x, edge_index, edge_attr)
    my_y: torch.Tensor = GlobalPool(reduce='sum')(my_out_lin(my_out))
    py_y: torch.Tensor = global_add_pool(
        py_out_lin(py_out), batch=torch.zeros(6, dtype=int))
    my_y.backward()
    py_y.backward()

    torch.testing.assert_close(my_out_lin.weight.grad, py_out_lin.weight.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_out_lin.bias.grad, py_out_lin.bias.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin1.weight.grad, py_lin1.weight.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin1.bias.grad, py_lin1.bias.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin2.weight.grad, py_lin2.weight.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(my_lin2.bias.grad, py_lin2.bias.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(myconv.lin.weight.grad, pyconv.lin.weight.grad,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(myconv.lin.bias.grad, pyconv.lin.bias.grad,
                               rtol=1e-6, atol=1e-6)
    assert not hasattr(myconv.eps, 'grad')
