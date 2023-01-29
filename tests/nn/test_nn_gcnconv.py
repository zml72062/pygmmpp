import sys
try:
    sys.path.append('.')
except:
    pass
import torch
import torch.nn
import torch.nn.functional as F
import math
from pygmmpp.nn.gcn_conv import GCNConv as myGCNConv
from pygmmpp.nn.pool import GlobalPool


def test_nn_gcnconv_forward():
    torch.manual_seed(452)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int)
    myconv = myGCNConv(10, 20)

    my_out = myconv(x, edge_index, 6)

    weight = myconv.linear.weight
    bias = myconv.linear.bias

    torch.testing.assert_close(my_out[0],
                               weight @ (
        (x[1]+x[3])/math.sqrt(12) + x[0]/3
    ) + bias, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(my_out[1],
                               weight @ (
        (x[0]+x[2])/math.sqrt(12) + (x[1]+x[4])/4
    ) + bias, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(my_out[2],
                               weight @ (
        (x[1]+x[3])/math.sqrt(12) + x[2]/3
    ) + bias, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(my_out[3],
                               weight @ (
        (x[0]+x[2])/math.sqrt(12) + (x[3]+x[4])/4
    ) + bias, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(my_out[4],
                               weight @ (
        (x[1]+x[3]+x[4])/4 + x[5]/math.sqrt(8)
    ) + bias, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(my_out[5],
                               weight @ (
        x[4]/math.sqrt(8) + x[5]/2
    ) + bias, atol=1e-6, rtol=1e-6)


def test_nn_gcnconv_backward():
    torch.manual_seed(32)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 3, 4, 1],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3, 0, 1, 4]], dtype=int)
    myconv = myGCNConv(10, 20)
    lin = torch.nn.Linear(20, 1, bias=False)

    my_out = myconv(x, edge_index, 6)
    model = torch.nn.Sequential(
        torch.nn.ReLU(),
        GlobalPool(reduce='sum'),
        lin
    )

    y: torch.Tensor = model(my_out)
    y.backward()

    def relu_derivative(x: torch.Tensor):
        t = torch.ones_like(x)
        t[x < 0] = 0
        return t

    torch.testing.assert_close(lin.weight.grad,
                               F.relu(my_out).sum(dim=0, keepdim=True),
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(myconv.linear.bias.grad,
                               relu_derivative(my_out).sum(
                                   dim=0) * lin.weight.reshape((20,)),
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(myconv.linear.weight.grad.T, (
        (x[1]+x[3])/math.sqrt(12) + x[0]/3
    ).view(-1, 1) @ (relu_derivative(my_out)[0] * lin.weight) + (
        (x[0]+x[2])/math.sqrt(12) + (x[1]+x[4])/4
    ).view(-1, 1) @ (relu_derivative(my_out)[1] * lin.weight) + (
        (x[1]+x[3])/math.sqrt(12) + x[2]/3
    ).view(-1, 1) @ (relu_derivative(my_out)[2] * lin.weight) + (
        (x[0]+x[2])/math.sqrt(12) + (x[3]+x[4])/4
    ).view(-1, 1) @ (relu_derivative(my_out)[3] * lin.weight) + (
        (x[1]+x[3]+x[4])/4 + x[5]/math.sqrt(8)
    ).view(-1, 1) @ (relu_derivative(my_out)[4] * lin.weight) + (
        x[4]/math.sqrt(8) + x[5]/2
    ).view(-1, 1) @ (relu_derivative(my_out)[5] * lin.weight),
        atol=1e-6, rtol=1e-6)
