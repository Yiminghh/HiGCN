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
import torch_sparse

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


class HIMnet_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Order=2, bias=True, **kwargs):
        super(HIMnet_prop, self).__init__(aggr='add', **kwargs)
        # K=10, alpha=0.1, Init='PPR',
        self.K = K
        self.alpha = alpha
        self.Order = Order
        #filterWeights = initFilterWeight(Init, alpha, K, Gamma)
        # fw已经在reset_parameters中初始化了，看看下面这边删掉有没有影响
        # filterWeights = alpha * (1 - alpha) ** np.arange(K + 1)
        # filterWeights[-1] = (1 - alpha) ** K
        # self.fW = Parameter(torch.tensor(filterWeights))
        self.fW = Parameter(torch.Tensor(self.K + 1))
        self.exp_a = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化参数
        """
        torch.nn.init.zeros_(self.fW)
        for k in range(self.K + 1):
            self.fW.data[k] = self.alpha * (1 - self.alpha) ** k
        self.fW.data[-1] = (1 - self.alpha) ** self.K

        torch.nn.init.ones_(self.exp_a)



    def forward(self, x, HL):
        EXP_A = F.relu(self.exp_a)#大于0的指数函数保证是递增函数

        # hidden = x * (self.fW[0])
        # for k in range(self.K):
        #     x = matmul(HL, x, reduce=self.aggr)
        #     gamma = self.fW[k + 1]
        #     hidden = hidden + gamma * x

        temp = [x, matmul(HL,x)]
        for k in range(2,self.K):
            temp.append(2*matmul(HL,temp[k-1]) - temp[k-2])

        #hidden=torch.zeros(x.size(), device=device)
        hidden = self.fW[0] * temp[0]
        for k in range(1, self.K):
            hidden = hidden + self.fW[k]/(k**EXP_A) * temp[k]

        #hidden = torch.stack(temp, dim=0)
        #hidden = torch.einsum('bij,jk->ik', hidden,t )
        self.beta = 0
        for k in range(1, self.K, 4):
            self.beta = self.beta + k*self.fW[k]/(k**EXP_A)
        for k in range(3, self.K, 4):
            self.beta = self.beta - k*self.fW[k]/(k**EXP_A)

        return hidden

    # 自定义打印结构
    def __repr__(self):
        return '{}(Order={}, K={}, exp_a={}, beta={}, filterWeights={})'.format(self.__class__.__name__, self.Order, self.K, self.exp_a, self.beta, self.fW)

    # 定义信息汇聚函数
    # 我们没有调用propagate函数，所以其实没有用到message函数
    # def message(self, x_j, norm):
    #     # 正则化
    #     # norm.view(-1,1)将norm变为一个列向量
    #     # x_j是节点的特征表示矩阵
    #     return norm.view(-1, 1) * x_j



class HIMnet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(HIMnet, self).__init__()
        self.Order = args.Order
        self.lin_in = nn.ModuleList()
        self.hgc = nn.ModuleList()

        for i in range(args.Order):
            self.lin_in.append(Linear(dataset.num_features, args.hidden))
            self.hgc.append(HIMnet_prop(args.K, args.alpha, args.Order))

        self.lin_out = Linear(args.hidden * args.Order, dataset.num_classes)


        #self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.hgc.reset_parameters()

    def forward(self, data):

        x, HL = data.x, data.HL
        x_concat = torch.tensor([]).to(device)
        for p in range(self.Order):
            xx = F.dropout(x, p=self.dropout, training=self.training)
            xx = self.lin_in[p](xx)
            if self.dprate > 0.0:
                xx = F.dropout(xx, p=self.dprate, training=self.training)
            xx = self.hgc[p](xx, HL[p + 1])
            x_concat = torch.concat((x_concat, xx), 1)

        x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
        x_concat = self.lin_out(x_concat)
        # x_concat = F.leaky_relu(x_concat)

        return F.log_softmax(x_concat, dim=1)  # 为什么要加log_softmax


