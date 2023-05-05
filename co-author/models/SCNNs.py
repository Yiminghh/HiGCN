import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np


def scnn_chebyshev_assemble(K, L, x):
    (B, C_in, M) = x.shape  # B 表示批次大小，C_in 表示输入通道数，M 表示特征的长度。
    assert (L.shape[0] == M)
    assert (L.shape[0] == L.shape[1])
    assert (K > 0)  # 卷积核阶数

    X = []
    for b in range(0, B):
        X123 = []
        for c_in in range(0, C_in):
            X23 = []
            X23.append(x[b, c_in, :].unsqueeze(1))  # Constant, k = 0 term.

            if K > 1:
                X23.append(L.mm(X23[0]))
            for k in range(2, K):
                # X23.append(2*(L.mm(X23[k-1])) - X23[k-2]) # original
                X23.append(L.mm(X23[k - 1]))  # changed by ms # 1
                # X23.append(L.mm(X23[0]) # 2
                # X23.append(X23[0])

            X23 = torch.cat(X23, 1)
            assert (X23.shape == (M, K))
            X123.append(X23.unsqueeze(0))

        X123 = torch.cat(X123, 0)
        assert (X123.shape == (C_in, M, K))
        X.append(X123.unsqueeze(0))

    X = torch.cat(X, 0)
    assert (X.shape == (B, C_in, M, K))

    return X

class SimplicialConvolution(nn.Module):
    def __init__(self, K, C_in, C_out, enable_bias=True, variance=1.0, groups=1):
        assert groups == 1, "Only groups = 1 is currently supported."
        super().__init__()

        assert (C_in > 0)
        assert (C_out > 0)
        assert (K > 0)

        self.C_in = C_in
        self.C_out = C_out
        self.K = K
        self.enable_bias = enable_bias

        self.theta = nn.parameter.Parameter(variance * torch.randn((self.C_out, self.C_in, self.K)))
        if self.enable_bias:
            self.bias = nn.parameter.Parameter(torch.zeros((1, self.C_out, 1)))
        else:
            self.bias = 0.0

    def forward(self, L, x):
        assert (len(L.shape) == 2)
        assert (L.shape[0] == L.shape[1])

        (B, C_in, M) = x.shape

        assert (M == L.shape[0])
        assert (C_in == self.C_in)

        X = scnn_chebyshev_assemble(self.K, L, x)
        y = torch.einsum("bimk,oik->bom", (X, self.theta))  # tensor multiplication
        assert (y.shape == (B, self.C_out, M))

        return y + self.bias


class SimplicialConvolution2(nn.Module):
    def __init__(self, K1, K2, C_in, C_out, enable_bias=True, variance=1.0, groups=1):
        assert groups == 1, "Only groups = 1 is currently supported."
        super().__init__()

        assert (C_in > 0)
        assert (C_out > 0)
        assert (K1 > 0)
        assert (K2 > 0)

        self.C_in = C_in
        self.C_out = C_out
        self.K1 = K1
        self.K2 = K2
        self.enable_bias = enable_bias

        self.theta1 = nn.parameter.Parameter(variance * torch.randn((self.C_out, self.C_in, self.K1 + self.K2)))
        if self.enable_bias:
            self.bias1 = nn.parameter.Parameter(torch.zeros((1, self.C_out, 1)))
        else:
            self.bias1 = 0.0

    def forward(self, Ll, Lu, x):
        assert (len(Ll.shape) == 2)
        assert (Ll.shape[0] == Ll.shape[1])
        assert (len(Lu.shape) == 2)
        assert (Lu.shape[0] == Lu.shape[1])
        (B, C_in, M) = x.shape

        assert (M == Ll.shape[0])
        assert (M == Lu.shape[0])
        assert (C_in == self.C_in)

        X1 = scnn_chebyshev_assemble(self.K1, Ll, x)
        X2 = scnn_chebyshev_assemble(self.K2, Lu, x)
        X = torch.cat((X1, X2), 3)
        assert (X.shape == (B, self.C_in, M, self.K1 + self.K2))
        y = torch.einsum("bimk,oik->bom", (X, self.theta1))
        y = y + self.bias1

        assert (y.shape == (B, self.C_out, M))

        return y


class SNN(nn.Module):
    def __init__(self, data, args):
        super().__init__()


        self.colors = data.xs[0].shape[1]
        assert (self.colors > 0)

        num_filters = 30 #20
        variance = 0.01 #0.001
        K = 5 # filter order
        # Degree 0 convolutions.
        self.C0_1 = SimplicialConvolution(K, self.colors, num_filters*self.colors, variance=variance)
        self.C0_2 = SimplicialConvolution(K, num_filters*self.colors, num_filters*self.colors, variance=variance)
        self.C0_3 = SimplicialConvolution(K, num_filters*self.colors, self.colors, variance=variance)


    def forward(self, data):
        Ls, Ds, adDs, xs = data.Ls, data.Ds, data.adDs, data.xs
        #assert(len(xs) == 6) # The three degrees are fed together as a list.

        assert(len(Ls) == len(Ds))
        Ms = [L.shape[0] for L in Ls]
        Ns = [D.shape[0] for D in Ds]

        Bs = [x.shape[0] for x in xs] #每个batch的大小，全1
        C_ins = [x.shape[1] for x in xs] #每个特征的维度，全1
        Ms = [x.shape[2] for x in xs]

        assert(Ms == [D.shape[1] for D in Ds])
        assert(Ms == [L.shape[1] for L in Ls])
        assert([adD.shape[0] for adD in adDs] == [D.shape[1] for D in Ds])
        assert([adD.shape[1] for adD in adDs] == [D.shape[0] for D in Ds])

        assert(Bs == len(Bs)*[Bs[0]])
        assert(C_ins == len(C_ins)*[C_ins[0]])

        out0_1 = self.C0_1(Ls[0], xs[0]) #+ self.D10_1(xs[1])
        out0_2 = self.C0_2(Ls[0], nn.LeakyReLU()(out0_1)) #+ self.D10_2(nn.LeakyReLU()(out1_1))
        out0_3 = self.C0_3(Ls[0], nn.LeakyReLU()(out0_2)) #+ self.D10_3(nn.LeakyReLU()(out1_2))


        return out0_3.view(-1)


class SCNN(nn.Module):
    def __init__(self, data, args):
        super().__init__()

        self.colors = data.xs[0].shape[1]
        assert (self.colors > 0)

        num_filters = 30 #20
        variance = 0.01 #0.001
        K1 = 1#2
        K2 = 2#3
        # Degree 0 convolutions.
        self.C0_1 = SimplicialConvolution2(K1, K2, self.colors, num_filters*self.colors, variance=variance)
        self.C0_2 = SimplicialConvolution2(K1, K2, num_filters*self.colors, num_filters*self.colors, variance=variance)
        self.C0_3 = SimplicialConvolution2(K1, K2, num_filters*self.colors, self.colors, variance=variance)

    def forward(self, data):
        Lls, Lus, Ds, adDs, xs = data.Lls, data.Lus, data.Ds, data.adDs, data.xs
        #assert(len(xs) == 6) # The three degrees are fed together as a list.

        assert(len(Lls) == len(Ds))
        assert(len(Lus) == len(Ds))
        Ms = [Ll.shape[0] for Ll in Lls]
        Ms = [Lu.shape[0] for Lu in Lus]
        Ns = [D.shape[0] for D in Ds]

        Bs = [x.shape[0] for x in xs]
        C_ins = [x.shape[1] for x in xs]
        Ms = [x.shape[2] for x in xs]

        assert(Ms == [D.shape[1] for D in Ds])
        assert(Ms == [Ll.shape[1] for Ll in Lls])
        assert(Ms == [Lu.shape[1] for Lu in Lus])
        assert([adD.shape[0] for adD in adDs] == [D.shape[1] for D in Ds])
        assert([adD.shape[1] for adD in adDs] == [D.shape[0] for D in Ds])

        assert(Bs == len(Bs)*[Bs[0]])
        assert(C_ins == len(C_ins)*[C_ins[0]])

        out0_1 = self.C0_1(Lls[0], Lus[0], xs[0]) #+ self.D10_1(xs[1])
        out0_2 = self.C0_2(Lls[0], Lus[0], nn.LeakyReLU()(out0_1)) #+ self.D10_2(nn.LeakyReLU()(out1_1))
        out0_3 = self.C0_3(Lls[0], Lus[0], nn.LeakyReLU()(out0_2)) #+ self.D10_3(nn.LeakyReLU()(out1_2))

        return out0_3.view(-1)