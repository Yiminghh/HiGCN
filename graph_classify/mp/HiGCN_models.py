


import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d as BN
from torch_geometric.nn import GINConv, JumpingKnowledge
from mp.nn import get_nonlinearity, get_pooling_fn
from torch_geometric.nn import GCNConv
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


class HiGCN_old(torch.nn.Module):
    def __init__(self, Order, num_features, K, hidden, num_classes, dropout=0.5, alpha=0.5,
                 readout='sum', nonlinearity='relu'):
        super(HiGCN_old, self).__init__()
        self.Order = Order
        self.lin_in = nn.ModuleList()
        self.hgc = nn.ModuleList()
        self.pooling_fn = get_pooling_fn(readout)
        self.nonlinearity = nonlinearity

        for i in range(Order):
            self.lin_in.append(Linear(num_features, hidden))
            self.hgc.append(HiGCN_prop(K, alpha, Order))

        self.lin_out1 = Linear(hidden * Order, hidden)
        self.lin_out2 = Linear(hidden, num_classes)

        self.dprate = dropout
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.lin_in:
            conv.reset_parameters()
        for conv in self.hgc:
            conv.reset_parameters()
        self.lin_out1.reset_parameters()
        self.lin_out2.reset_parameters()

    def forward(self, data):
        x, HL, batch = data.x, data.HL, data.batch
        model_nonlinearity = get_nonlinearity(self.nonlinearity, return_module=False)
        x_concat = torch.tensor([]).to(device)

        for i in range(self.Order):
            xx = F.dropout(x, p=self.dropout, training=self.training)
            xx = self.lin_in[i](xx)
            if self.dprate > 0.0:
                xx = F.dropout(xx, p=self.dprate, training=self.training)
            xx = self.hgc[i](xx, HL[i + 1])
            #xx = self.pooling_fn(xx, batch)  # readout
            x_concat = torch.concat((x_concat, xx), 1)

        x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)


        x_concat = self.pooling_fn(x_concat, batch)  # readout
        x_concat = self.lin_out1(x_concat)
        x_concat = model_nonlinearity(x_concat)
        x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
        x_concat = self.lin_out2(x_concat)

        return x_concat

    def __repr__(self):
        return self.__class__.__name__




class HiGCN_layer(torch.nn.Module):
    def __init__(self, Order, num_features, K, hidden, num_classes, dropout=0.5, alpha=0.5,
                 readout='sum', nonlinearity='relu'):
        super(HiGCN_layer, self).__init__()
        self.Order = Order
        self.lin_in = nn.ModuleList()
        self.hgc = nn.ModuleList()
        self.pooling_fn = get_pooling_fn(readout)
        self.nonlinearity = nonlinearity

        for i in range(Order):
            self.lin_in.append(Linear(num_features, hidden))
            self.hgc.append(HiGCN_prop(K, alpha, Order))

        self.lin_out = Linear(hidden * Order, hidden)


        self.dprate = dropout
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.lin_in:
            conv.register_parameter()
        for conv in self.hgc:
            conv.register_parameter()
        self.lin_out.reset_parameters()


    def forward(self, data):
        x, HL, batch = data.x, data.HL, data.batch
        model_nonlinearity = get_nonlinearity(self.nonlinearity, return_module=False)
        x_concat = torch.tensor([]).to(device)

        for i in range(self.Order):
            xx = F.dropout(x, p=self.dropout, training=self.training)
            xx = self.lin_in[i](xx)
            if self.dprate > 0.0:
                xx = F.dropout(xx, p=self.dprate, training=self.training)
            xx = self.hgc[i](xx, HL[i + 1])
            x_concat = torch.concat((x_concat, xx), 1)

        x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)


        x_concat = self.pooling_fn(x_concat, batch)  # readout
        x_concat = self.lin_out1(x_concat)
        x_concat = model_nonlinearity(x_concat)
        x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
        x_concat = self.lin_out2(x_concat)

        return x_concat

    def __repr__(self):
        return self.__class__.__name__










