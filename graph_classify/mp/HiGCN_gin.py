import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP
from torch_sparse import matmul, SparseTensor
from mp.nn import get_nonlinearity, get_pooling_fn

device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch.nn import Linear, Sequential, BatchNorm1d as BN


class HiGCN_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Order=2, bias=True, **kwargs):
        super(HiGCN_prop, self).__init__(aggr='add', **kwargs)
        # K=10, alpha=0.1, Init='PPR',
        self.K = K
        self.alpha = alpha
        self.Order = Order
        self.fW = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.fW)
        for k in range(self.K + 1):
            self.fW.data[k] = self.alpha * (1 - self.alpha) ** k
        self.fW.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, HL):

        hidden = x * (self.fW[0])
        for k in range(self.K):
            x = matmul(HL, x, reduce=self.aggr)
            gamma = self.fW[k + 1]
            hidden = hidden + gamma * x

        return hidden

    # 自定义打印结构
    def __repr__(self):
        return '{}(Order={}, K={}, filterWeights={})'.format(self.__class__.__name__, self.Order, self.K, self.fW)










class HiGCNConv(MessagePassing):

    def __init__(self, num_features, order, hidden, num_classess, dropout_rate, nn, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn

        self.dropout = dropout_rate
        self.dprate = dropout_rate
        self.Order = order

        self.lin_in = torch.nn.ModuleList()
        self.hgc = torch.nn.ModuleList()
        for i in range(self.Order):
            self.lin_in.append(Linear(num_features, hidden))
            self.hgc.append(HiGCN_prop(K=10, alpha=0.5, Order=self.Order))
        self.lin_out = Linear(hidden * self.Order, num_classess)
        self.reset_parameters()


    def reset_parameters(self):
        self.lin_out.reset_parameters()
        for conv in self.lin_in:
            conv.reset_parameters()
        for conv in self.hgc:
            conv.reset_parameters()

    def forward(self, x, HL) :
        x_concat = torch.tensor([]).to(device)

        for i in range(self.Order):
            xx = F.dropout(x, p=self.dropout, training=self.training)
            xx = self.lin_in[i](xx)
            if self.dprate > 0.0:
                xx = F.dropout(xx, p=self.dprate, training=self.training)
            xx = self.hgc[i](xx, HL[i + 1])
            x_concat = torch.concat((x_concat, xx), 1)

        x_concat = self.lin_out(x_concat)
        return self.nn(x_concat)



    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'



class HiGCN(torch.nn.Module):
    def __init__(self, max_petal_dim, num_features, num_layers, hidden, num_classes, readout='sum',
                 dropout_rate=0.5, nonlinearity='relu'):
        super(HiGCN, self).__init__()
        self.order = max_petal_dim
        self.pooling_fn = get_pooling_fn(readout)
        self.nonlinearity = nonlinearity
        self.dropout_rate = dropout_rate
        conv_nonlinearity = get_nonlinearity(nonlinearity, return_module=True)
        self.conv1 = HiGCNConv(num_features, self.order, hidden, hidden, dropout_rate,
            Sequential(
                Linear(hidden, hidden),
                BN(hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                BN(hidden),
                conv_nonlinearity(),
            ))
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                HiGCNConv(hidden, self.order, hidden, hidden, dropout_rate,
                    Sequential(
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                    ))
            )
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        model_nonlinearity = get_nonlinearity(self.nonlinearity, return_module=False)
        x, HL, batch = data.x, data.HL, data.batch
        x = self.conv1(x, HL)
        for conv in self.convs:
            x = conv(x, HL)
        x = self.pooling_fn(x, batch)#readout
        x = model_nonlinearity(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__