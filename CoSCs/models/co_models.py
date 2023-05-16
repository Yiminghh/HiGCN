import copy

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class co_HiGCN(torch.nn.Module):
    def __init__(self,data, args):
        super(co_HiGCN, self).__init__()
        self.Order = args.Order
        self.lin_in = nn.ModuleList()
        self.hgc = nn.ModuleList()
        self.in_dim = data.x.shape[1]
        self.out_dim = data.y.shape[1]

        for i in range(args.Order):
            self.lin_in.append(Linear( self.in_dim, args.hidden))
            self.hgc.append(HiGCN_prop(args.K, args.alpha, args.Order))

        self.lin_out = Linear(args.hidden * args.Order, self.out_dim)
        self.lin_out1 = Linear(args.hidden * args.Order, args.hidden)
        self.lin_out2 = Linear(args.hidden, self.out_dim)


        #self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.hgc:
            layer.reset_parameters()
        for layer in self.lin_in:
            layer.reset_parameters()
        self.lin_out1.reset_parameters()
        self.lin_out2.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(self, data):

        x, HL = data.x, data.HL
        x_concat = torch.tensor([]).to(device)
        for i in range(self.Order):
            xx = F.dropout(x, p=self.dropout, training=self.training)
            xx = self.lin_in[i](xx)
            if self.dprate > 0.0:
                xx = F.dropout(xx, p=self.dprate, training=self.training)
            xx = self.hgc[i](xx, HL[i + 1])
            #xx = F.leaky_relu(xx)
            x_concat = torch.concat((x_concat, xx), 1)


        x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)

        x_concat = self.lin_out1(x_concat)
        #x_concat = F.tanh(x_concat)
        #x_concat = F.sigmoid(x_concat)
        x_concat = F.leaky_relu(x_concat)
        x_concat = self.lin_out2(x_concat)
        x_concat = F.leaky_relu(x_concat)

        return x_concat  # 为什么要加log_softmax


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
