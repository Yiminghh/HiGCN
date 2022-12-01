
import torch
import numpy as np
import networkx as nx
from torch_sparse import SparseTensor, fill_diag, matmul, mul


def creat_L_SparseTensor(G: nx.Graph, maxCliqueSize=2):
    """

    Parameters
    ----------
    G
    maxCliqueSize

    Returns
    -------

    """
    if maxCliqueSize == 2:
        return creat_L2(G)

    itClique = nx.enumerate_all_cliques(G)
    N = np.max(G.nodes) + 1

    nextClique = next(itClique)
    # 跳过所有0团（节点）
    while len(nextClique) <= 1:
        nextClique = next(itClique)

    # 数据初始化
    _row, _col, _temp, _D = {}, {}, {}, {}
    for i in range(1, maxCliqueSize + 1):
        _row[i], _col[i] = [], []
        _temp[i] = np.ones(i + 1, dtype=np.int)
        _D[i] = np.zeros(N)

    cliqueNum = np.zeros(maxCliqueSize + 1, dtype=int)  # 记录各阶团的数量
    # 寻找所有的高阶结构
    while len(nextClique) <= maxCliqueSize + 1:
        order = len(nextClique) - 1

        _row[order].extend(nextClique)
        _col[order].extend(cliqueNum[order] * _temp[order])
        for node in nextClique:
            _D[order][node] += 1
        cliqueNum[order] += 1
        # 寻找迭代器的下一个团
        try:
            nextClique = next(itClique)
        except StopIteration:
            break

    # 构造高阶关联矩阵H,和高阶Laplacian矩阵L
    H, L = {}, {}
    for k in range(1, maxCliqueSize + 1):
        H[k] = torch.sparse_coo_tensor(torch.LongTensor((_row[k], _col[k])), torch.ones(len(_row[k])),
                                       [G.number_of_nodes(), cliqueNum[k]])
        H[k] = SparseTensor.from_torch_sparse_coo_tensor(H[k])

        D_inv_sqrt = torch.tensor(_D[k]).pow_(-0.5)
        D_inv_sqrt.masked_fill_(D_inv_sqrt == float('inf'), 0.)
        L[k] = matmul(H[k], H[k].t())
        L[k] = mul(L[k], D_inv_sqrt.view(1, -1))
        L[k] = mul(L[k], D_inv_sqrt.view(-1, 1) / (k + 1))

    print("n0:{}, n1:{}, n2:{}".format(G.number_of_nodes(), G.number_of_edges(), sum(_D[2]) / 3))
    return L

def creat_L2(G: nx.Graph):
    maxCliqueSize = 2
    larger_nei = {}
    for v in G.nodes:
        larger_nei[v] = {nei for nei in G.neighbors(v) if nei > v}

    _D, L = {}, {}
    for i in range(1, maxCliqueSize + 1):
        _D[i] = torch.zeros(G.number_of_nodes())
        L[i] = torch.zeros((G.number_of_nodes(), G.number_of_nodes()))

    for edge in G.edges:
        print(edge)
        a = edge[0]
        b = edge[1]
        if a==b:
            continue
        # 1阶
        _D[1][a] += 1
        _D[1][b] += 1
        L[1][a, a] += 1
        L[1][b, b] += 1
        L[1][a, b] += 1
        L[1][b, a] += 1
        # 2阶
        com_nei = larger_nei[a] & larger_nei[b]
        _D[2][a] += len(com_nei)
        _D[2][b] += len(com_nei)
        L[2][a, a] += len(com_nei)
        L[2][b, b] += len(com_nei)
        L[2][a, b] += len(com_nei)
        L[2][b, a] += len(com_nei)

        for i in com_nei:
            L[2][a, i] += 1
            L[2][i, a] += 1
            L[2][b, i] += 1
            L[2][i, b] += 1
            L[2][i, i] += 1
            _D[2][i] += 1
            print("三角形：",a,b,i)
    print("n0:{}, n1:{}, n2:{}".format(G.number_of_nodes(), G.number_of_edges(),sum(_D[2])/3))
    new_triangle = sum(nx.triangles(G).values()) / 3
    print("总共三角形数：",new_triangle)

    for k_ in range(1, maxCliqueSize + 1):
        L[k_] = SparseTensor.from_dense(L[k_])
        D_inv_sqrt = _D[k_].pow_(-0.5)
        D_inv_sqrt.masked_fill_(D_inv_sqrt == float('inf'), 0.)
        L[k_] = mul(L[k_], D_inv_sqrt.view(1, -1))
        L[k_] = mul(L[k_], D_inv_sqrt.view(-1, 1) / (k_ + 1))

    return L