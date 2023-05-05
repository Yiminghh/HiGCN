import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from torch_geometric.nn import MessagePassing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class HiSCN_layer_pandemic(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init,  Order=2, Gamma=None, bias=True, **kwargs):
        super(HiSCN_layer_pandemic, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Order = Order
        filterWeights = initFilterWeight(Init, alpha, K, Gamma)
        self.fW = Parameter(torch.tensor(filterWeights))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.fW)
        for k in range(self.K + 1):
            self.fW.data[k] = self.alpha * (1 - self.alpha) ** k
        self.fW.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, HL):

        hidden = x * (self.fW[0])
        for k in range(self.K):
            x = torch.matmul(HL, x)
            gamma = self.fW[k + 1]
            hidden = hidden + gamma * x

        return hidden

    # 自定义打印结构
    def __repr__(self):
        return '{}(Order={}, K={}, filterWeights={})'.format(self.__class__.__name__, self.Order, self.K, self.fW)


class HiSCN_pandemic(torch.nn.Module):
    def __init__(self, Order, K, alpha, Init, dprate, dropout, dim_in, dim_hid, dim_out):
        super(HiSCN_pandemic, self).__init__()
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        self.Order = Order
        self.K = K
        self.alpha = alpha
        self.Init = Init
        self.dprate = dropout
        self.dropout = dropout

        self.lin_in = nn.ModuleList()
        self.hgc = nn.ModuleList()

        for i in range(self.Order):
            self.lin_in.append(Linear(self.dim_in, self.dim_hid))
            self.hgc.append(HiSCN_layer_pandemic(self.K, self.alpha, self.Init, self.Order))

        #self.lin_out = Linear(self.dim_hid * self.Order, self.dim_out)

    def reset_parameters(self):
        self.hgc.reset_parameters()

    def forward(self, x, laplacian):

        x_concat = torch.tensor([]).to(device)
        for i in range(self.Order):
            xx = F.dropout(x, p=self.dropout, training=self.training)
            xx = self.lin_in[i](xx)
            # xx = F.relu(xx)   #add
            if self.dprate > 0.0:
                xx = F.dropout(xx, p=self.dprate, training=self.training)
            xx = self.hgc[i](xx, laplacian[i + 1])
            x_concat = torch.concat((x_concat, xx), 2)

        # x_concat = F.relu(x_concat) #add
        #x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
        #x_concat = self.lin_out(x_concat)

        return x_concat


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.act = nn.ELU()
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return self.act(output + self.bias)
        else:
            return self.act(output)


class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, tanhalpha=1):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding):
        # embedding [batchsize, hidden_dim]
        nodevec1 = self.linear1(embedding)
        nodevec2 = self.linear2(embedding)
        nodevec1 = self.alpha * nodevec1
        nodevec2 = self.alpha * nodevec2
        nodevec1 = torch.tanh(nodevec1)
        nodevec2 = torch.tanh(nodevec2)

        adj = torch.bmm(nodevec1, nodevec2.permute(0, 2, 1)) - torch.bmm(nodevec2, nodevec1.permute(0, 2, 1))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj


class ConvBranch(nn.Module):
    def __init__(self,
                 m,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation_factor,
                 hidP=1,
                 isPool=True):
        super().__init__()
        self.m = m
        self.isPool = isPool
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), dilation=(dilation_factor, 1))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        if self.isPool:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))
        # self.activate = nn.Tanh()

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.isPool:
            x = self.pooling(x)
        x = x.view(batch_size, -1, self.m)
        return x


class RegionAwareConv(nn.Module):
    def __init__(self, P, m, k, hidP, dilation_factor=2):
        super(RegionAwareConv, self).__init__()
        self.P = P
        self.m = m
        self.k = k
        self.hidP = hidP
        self.conv_l1 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=3, dilation_factor=1,
                                  hidP=self.hidP)
        self.conv_l2 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=5, dilation_factor=1,
                                  hidP=self.hidP)
        self.conv_p1 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=3,
                                  dilation_factor=dilation_factor, hidP=self.hidP)
        self.conv_p2 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=5,
                                  dilation_factor=dilation_factor, hidP=self.hidP)
        self.conv_g = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=self.P, dilation_factor=1,
                                 hidP=None, isPool=False)
        self.activate = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, 1, self.P, self.m)
        batch_size = x.shape[0]
        # local pattern
        x_l1 = self.conv_l1(x)
        x_l2 = self.conv_l2(x)
        x_local = torch.cat([x_l1, x_l2], dim=1)
        # periodic pattern
        x_p1 = self.conv_p1(x)
        x_p2 = self.conv_p2(x)
        x_period = torch.cat([x_p1, x_p2], dim=1)
        # global
        x_global = self.conv_g(x)
        # concat and activate
        x = torch.cat([x_local, x_period, x_global], dim=1).permute(0, 2, 1)
        x = self.activate(x)
        return x
