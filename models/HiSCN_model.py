#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 这个函数没有用了发布时候再删掉
def initFilterWeight(Init, alpha, K, Gamma=None):
    """
    构造滤波器的多项式系数
    """
    assert Init in ['SGC', 'PPR', 'NPPR', 'Random',
                    'WS'], 'The method for initialising the filter weights is not defined.'
    if Init == 'SGC':
        # note that in SCG model, alpha has to be a integer.
        # It means where the peak at when initializing GPR weights.
        filterWeights = 0.0 * np.ones(K + 1)
        filterWeights[alpha] = 1.0
    elif Init == 'PPR':
        # PPR-like
        filterWeights = alpha * (1 - alpha) ** np.arange(K + 1)
        filterWeights[-1] = (1 - alpha) ** K
    elif Init == 'NPPR':
        # Negative PPR
        filterWeights = (alpha) ** np.arange(K + 1)
        filterWeights = filterWeights / np.sum(np.abs(filterWeights))
    elif Init == 'Random':
        # Random
        bound = np.sqrt(3 / (K + 1))
        filterWeights = np.random.uniform(-bound, bound, K + 1)
        filterWeights = filterWeights / np.sum(np.abs(filterWeights))
    elif Init == 'WS':
        # Specify Gamma
        # 指定的Gamma
        filterWeights = Gamma
    return filterWeights


class HiSCN_layer(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Order=2, bias=True, **kwargs):
        super(HiSCN_layer, self).__init__(aggr='add', **kwargs)
        # K=10, alpha=0.1, Init='PPR',
        self.K = K
        #self.Init = Init
        self.alpha = alpha
        self.Order = Order
        #filterWeights = initFilterWeight(Init, alpha, K, Gamma)
        # filterWeights = alpha * (1 - alpha) ** np.arange(K + 1)
        # filterWeights[-1] = (1 - alpha) ** K
        # self.fW = Parameter(torch.tensor(filterWeights))
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


class HiSCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(HiSCN, self).__init__()
        self.Order = args.Order
        self.lin_in = nn.ModuleList()
        self.hgc = nn.ModuleList()

        for i in range(args.Order):
            self.lin_in.append(Linear(dataset.num_features, args.hidden))
            self.hgc.append(HiSCN_layer(args.K, args.alpha, args.Order))

        self.lin_out = Linear(args.hidden * args.Order, dataset.num_classes)


        #self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.hgc.reset_parameters()

    def forward(self, data):

        x, HL = data.x, data.HL
        x_concat = torch.tensor([]).to(device)
        for i in range(self.Order):
            xx = F.dropout(x, p=self.dropout, training=self.training)
            xx = self.lin_in[i](xx)
            if self.dprate > 0.0:
                xx = F.dropout(xx, p=self.dprate, training=self.training)
            xx = self.hgc[i](xx, HL[i + 1])
            x_concat = torch.concat((x_concat, xx), 1)

        x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
        x_concat = self.lin_out(x_concat)
        # x_concat = F.leaky_relu(x_concat)

        return F.log_softmax(x_concat, dim=1)  # 为什么要加log_softmax


