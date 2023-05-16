import torch
from torch_sparse import SparseTensor, mul
import itertools


def gen_HL_from_SC_list(number_of_nodes, simplex_list):
    """
    处理hubOrder=0节点级任务
    这个函数应该已经做到了空间占用最小,但不是很快
    本函数是HIMnet中的gen_HL_samll_space函数的简化版本
    """
    hubOrder = 0
    petalOrder = len(simplex_list[0])

    # 数据初始化
    n_core = number_of_nodes
    """
    _D:节点所连接的的三角形数
    _delta: 三角形中所包含的节点数，取了倒数
    """
    HL = torch.zeros(n_core, n_core)
    _D = torch.zeros(n_core)
    _delta = torch.ones((petalOrder + 1) ** 2)

    for simplex in simplex_list:
        print(simplex)
        order = len(simplex) - 1
        _D[simplex] += 1
        comb = torch.LongTensor([[a, b] for a, b in itertools.combinations(simplex, 2)])
        rows = torch.cat([torch.LongTensor(simplex), comb[:, 0], comb[:, 1]], dim=0)
        cols = torch.cat([torch.LongTensor(simplex), comb[:, 1], comb[:, 0]], dim=0)
        HL[rows, cols] += 1


    d_inv = _D
    d_inv[d_inv == 0] = 1
    d_inv = d_inv.pow_(-0.5)
    HL = SparseTensor.from_dense(HL)
    HL = mul(HL, d_inv.view(1, -1))
    HL = mul(HL, d_inv.view(-1, 1) / (petalOrder + 1))

    return HL