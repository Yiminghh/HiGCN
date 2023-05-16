import math

import torch
import numpy as np
import networkx as nx
from torch_sparse import SparseTensor, fill_diag, matmul, mul
import os
import yaml
import itertools
import pickle
import time

device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

def gen_HL_old(name, root, graph_list, petalType, petal_dim):
    """
    弃用，更新后函数为gen_HL
    """

    if petalType == "simplex":
        HL_path = os.path.join(root, name, name+'_HL.pt')
        max_petal_dim = 6 # 数据处理的时候把max_petal_dim设置大一点后面每次就不用重复计算了,原来是3
    elif petalType == "ring":
        HL_path = os.path.join(root, name, name + '_ring_HL.pt')
        max_petal_dim = 9 # 数据处理的时候把max_petal_dim设置大一点后面每次就不用重复计算了


    if os.path.exists(HL_path):
        HL = torch.load(HL_path)
        print("Load HL!")
    else:
        HL = []
        cliqueNum = np.zeros(max_petal_dim + 1, dtype=int)  # 记录各阶团的数量
        ringNum = np.zeros(max_petal_dim + 1, dtype=int)
        for idx, graph in enumerate(graph_list):
            print(f"generate HL for graph {idx}({graph.x.shape[0]} nodes and {graph.edge_index.shape[1]/2} edges)")
            src, dst = graph.edge_index
            edges = zip(src.tolist(), dst.tolist())
            G = nx.from_edgelist(edges)
            G.add_nodes_from(range(graph.x.shape[0])) #加这行代码因为可能存在孤立节点
            cells = {_: [] for _ in range(1, max_petal_dim + 1)}

            if petalType == 'simplex':
                # 找到所有的单纯形
                cliqueNum[0] += G.number_of_nodes()
                itClique = nx.enumerate_all_cliques(G)
                nextClique = next(itClique)
                # 跳过所有0团（节点）
                while len(nextClique) <= 1:
                    nextClique = next(itClique)
                while len(nextClique) <= max_petal_dim + 1:
                    order = len(nextClique) - 1
                    cells[order].append(sorted(nextClique))
                    cliqueNum[order] += 1
                    # 寻找迭代器的下一个团
                    try:
                        nextClique = next(itClique)
                    except StopIteration:
                        break
            elif petalType == 'ring':
                # 找 induced cycles
                ringNum[0] += G.number_of_nodes()
                graph = nx.DiGraph(G)

                def is_cycle_edge(i1, i2, cycle):
                    if i2 == i1 + 1:
                        return True
                    if i1 == 0 and i2 == len(cycle) - 1:
                        return True
                    return False

                def is_chordless(cycle):
                    for (i1, v1), (i2, v2) in itertools.combinations(enumerate(cycle), 2):
                        if not is_cycle_edge(i1, i2, cycle) and graph.has_edge(v1, v2):
                            return False
                    return True

                for cycle in nx.simple_cycles(graph):
                    # Because we need to use a DiGraph for this method, it will also return each edge
                    # as a cycle. So we skip these together with cycles above the maximum length.
                    if len(cycle) <= 1 or (len(cycle) > max_petal_dim + 1):
                        continue
                    # We skip the cycles with chords
                    if not is_chordless(cycle):
                        continue
                    # Store the cycle in a canonical form
                    order = len(cycle) - 1
                    cells[order].append(list(sorted(cycle)))
                    ringNum[order] += 1

                #for c in nx.cycle_basis(G):
                    # ringNum[len(c)-1] += 1
                    # if len(c) <= max_petal_dim:
                    #     cells[len(c)-1].append(list(sorted(c)))
                # NOTE: nx.cycle_basis(G)不会找到边，但nx.simple_cycles(nx.DiGraph(G))会，但执行很慢
                # 把边加入cell中，为了统一，order定义为圈中节点数-1
                # for edges in G.edges():
                #     cells[1].append(list(sorted(edges)))

            L = {}
            coreC_list = list(range(G.number_of_nodes()))

            if petalType == 'simplex':
                for order in range(1, max_petal_dim + 1):
                    L[order] = creat_HL_from_CWcomplex(coreC_list=coreC_list, petalC_list=cells[order])
            elif petalType == 'ring':
                for order in range(1, max_petal_dim + 1):
                    L[order] = creat_HL_from_CWcomplex(coreC_list=coreC_list, petalC_list=cells[order])


            HL.append(L)
            # L_old = creat_L_SparseTensor(G, maxCliqueSize=max_petal_dim)
            # torch.max(torch.abs(L[1].to_dense()-L_old[1].to_dense()))
        torch.save(HL, HL_path)
        # 记录图的统计信息
        if petalType == 'simplex':
            with open(os.path.join(root, name, name+'_simplex_statics.yaml'), "w") as f:
                statistics = {'graph_num': idx+1,
                              'n_simplex': cliqueNum.tolist()}
                yaml.dump(statistics, f)
        elif petalType == 'ring':
            with open(os.path.join(root, name, name + '_cell_statics.yaml'), "w") as f:
                statistics = {'graph_num': idx+1,
                              'n_rings': ringNum.tolist()}
                yaml.dump(statistics, f)



    #移除多余的HL
    # max_dim = min(max_petal_dim+1, max(HL[0].keys()))
    # for idx in range(len(graph_list)):
    #     for order in range(petal_dim+1, max_dim):
    #         HL[idx].pop(order)
            #del HL[idx][order]

    for idx in range(len(graph_list)):
        graph_list[idx].HL = {order: HL[idx][order] for order in range(1, petal_dim+1)}

    if isinstance(graph_list[0].x, torch.LongTensor):
        for idx in range(len(graph_list)):
            graph_list[idx].x = graph_list[idx].x.float()

    return graph_list

def gen_HL(name, root, graph_list, petalType, petal_dim):
    """
    petalType: simplex, ring
    petal_dim: 单纯形的时候对应的是其阶数，如=3表示四面体；ring时表示圈的大小-1
    max_petal_dim: 找单纯形和cycle时限制的阶数（初始找大一点后面就不用再找了）
    """

    if petalType == "simplex":
        HL_path = os.path.join(root, name, name+'_HL.pt')
        max_petal_dim = 6 # 数据处理的时候把max_petal_dim设置大一点后面每次就不用重复计算了,原来是3
    elif petalType == "ring":
        HL_path = os.path.join(root, name, name + '_ring_HL.pt')
        max_petal_dim = 9 # 数据处理的时候把max_petal_dim设置大一点后面每次就不用重复计算了


    if os.path.exists(HL_path):
        HL = torch.load(HL_path)
        print("Load HL!")
    else:
        HL = []
        tot_petal_num = np.zeros(max_petal_dim + 1, dtype=int)  # 记录各阶团的数量

        for idx, graph in enumerate(graph_list):
            print(f"generate HL for graph {idx}({graph.x.shape[0]} nodes and {graph.edge_index.shape[1]/2} edges)")
            src, dst = graph.edge_index
            edges = zip(src.tolist(), dst.tolist())
            G = nx.from_edgelist(edges)
            G.add_nodes_from(range(graph.x.shape[0])) #加这行代码因为可能存在孤立节点

            L, petalNum = gen_HL_samll_space(G, max_petal_dim=max_petal_dim, petalType=petalType)
            tot_petal_num = tot_petal_num + petalNum

            HL.append(L)

        torch.save(HL, HL_path)
        # 记录图的统计信息
        if petalType == 'simplex':
            with open(os.path.join(root, name, name+'_simplex_statics.yaml'), "w") as f:
                statistics = {'graph_num': idx+1,
                              'n_simplex': tot_petal_num.tolist()}
                yaml.dump(statistics, f)
        elif petalType == 'ring':
            with open(os.path.join(root, name, name + '_cell_statics.yaml'), "w") as f:
                statistics = {'graph_num': idx+1,
                              'n_rings': tot_petal_num.tolist()}
                yaml.dump(statistics, f)



    #移除多余的HL
    # max_dim = min(max_petal_dim+1, max(HL[0].keys()))
    # for idx in range(len(graph_list)):
    #     for order in range(petal_dim+1, max_dim):
    #         HL[idx].pop(order)
            #del HL[idx][order]

    for idx in range(len(graph_list)):
        graph_list[idx].HL = {order: HL[idx][order] for order in range(1, petal_dim+1)}

    if isinstance(graph_list[0].x, torch.LongTensor):
        for idx in range(len(graph_list)):
            graph_list[idx].x = graph_list[idx].x.float()

    return graph_list






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
    """
    这个代码更快
    TODO: 后面把所有代码都更新一下
    """
    maxCliqueSize = 2
    larger_nei = {}
    for v in G.nodes:
        larger_nei[v] = {nei for nei in G.neighbors(v) if nei > v}

    _D, L = {}, {}
    for i in range(1, maxCliqueSize + 1):
        _D[i] = torch.zeros(G.number_of_nodes())
        L[i] = torch.zeros((G.number_of_nodes(), G.number_of_nodes()))

    for edge in G.edges:
        #print(edge)
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
            #print("三角形：",a,b,i)
    #print("n0:{}, n1:{}, n2:{}".format(G.number_of_nodes(), G.number_of_edges(),sum(_D[2])/3))
    new_triangle = sum(nx.triangles(G).values()) / 3
    #print("总共三角形数：",new_triangle)

    for k_ in range(1, maxCliqueSize + 1):
        L[k_] = SparseTensor.from_dense(L[k_])
        D_inv_sqrt = _D[k_].pow_(-0.5)
        D_inv_sqrt.masked_fill_(D_inv_sqrt == float('inf'), 0.)
        L[k_] = mul(L[k_], D_inv_sqrt.view(1, -1))
        L[k_] = mul(L[k_], D_inv_sqrt.view(-1, 1) / (k_ + 1))

    return L, new_triangle


def creat_HL_from_CWcomplex(coreC_list, petalC_list):
    """
                     弃用
    coreC_list:
    petalC_list: cell的列表(圈或单纯形的列表）,只包含一种类型的cell
    universal_HL_from_simplexlist(name, hubOrder, maxOrder)能够处理任意hubOrder,maxOrder
    creat_HL_from_complex(name, coreC_list, petalC_list)要求花心只能是节点

    G = nx.read_edgelist(get_net_path(name), create_using=nx.Graph(), nodetype=int)
    L = creat_HL_from_complex(name, coreC_list=list(range(G.number_of_nodes())),petalC_list=get_simplex_list(name, maxOrder))
    """
    nc = len(coreC_list)  # 花心的规模（节点数）
    np = len(petalC_list)  # cell数量
    if np <= 0:
        return SparseTensor(row=torch.LongTensor([0]), col=torch.LongTensor([0]), value=torch.tensor([0]), sparse_sizes=(nc, nc))
    len_cell = len(petalC_list[0])  # 圈的长度

    _D = torch.zeros(nc)
    H = SparseTensor(row=torch.LongTensor([]), col=torch.LongTensor([]), value=torch.tensor([]),
                     sparse_sizes=(nc, np))

    ones_ = torch.ones(len_cell)
    for id, cell in enumerate(petalC_list):
        _D[cell] += 1
        H = H + SparseTensor(row=torch.LongTensor(cell), col=ones_.long() * id, value=ones_, sparse_sizes=(nc, np))

    A = H.matmul(H.t())
    _D[_D == 0] = 1
    _D = _D.pow_(-0.5)
    HL = mul(A, _D.view(1, -1))
    HL = mul(HL, _D.view(-1, 1) / len_cell)
    return HL



def gen_HL_samll_space_sparse(G: nx.graph, max_petal_dim=2, petalType='simplex'):
    """
    程序中使用的是 gen_HL_samll_space, 速度明显变快
    该函相比较于gen_HL_samll_space用了稀疏矩阵存储L,但是分子图都较小没必要
    """
    hubOrder=0
    def get_simplex(G:nx.graph, max_petal_dim):
        """ 生成所有dim<=max_petal_dim的simplex """
        itClique = nx.enumerate_all_cliques(G)
        nextClique = next(itClique)
        # 跳过所有0团（节点）
        while len(nextClique) <= 1:
            nextClique = next(itClique)
        while len(nextClique) <= max_petal_dim + 1:
            yield sorted(nextClique)
            # 寻找迭代器的下一个团
            try:
                nextClique = next(itClique)
            except StopIteration:
                break

    def get_induced_cycles(G:nx.graph, max_petal_dim):
        """Extracts the induced cycles from a graph using networkx."""
        graph = nx.DiGraph(G)
        max_ring = max_petal_dim + 1

        def is_cycle_edge(i1, i2, cycle):
            if i2 == i1 + 1:
                return True
            if i1 == 0 and i2 == len(cycle) - 1:
                return True
            return False

        def is_chordless(cycle):
            for (i1, v1), (i2, v2) in itertools.combinations(enumerate(cycle), 2):
                if not is_cycle_edge(i1, i2, cycle) and graph.has_edge(v1, v2):
                    return False
            return True

        nx_rings = set()
        for cycle in nx.simple_cycles(graph):
            # Because we need to use a DiGraph for this method, it will also return each edge
            # as a cycle. So we skip these together with cycles above the maximum length.
            if len(cycle) <= 1 or len(cycle) > max_ring:
                continue
            # We skip the cycles with chords
            if not is_chordless(cycle):
                continue

            yield sorted(cycle)

    if petalType == 'simplex':
        get_cells = get_simplex
    elif petalType == 'ring':
        get_cells = get_induced_cycles
    else:
        raise NotImplementedError

    # 数据初始化
    petalNum = np.zeros(max_petal_dim + 1, dtype=int)  # 记录各阶团的数量
    petalNum[0] = n_core = G.number_of_nodes()
    """
    _D:节点所连接的的三角形数
    _delta: 三角形中所包含的节点数，取了倒数
    """
    HL, _D, _delta = {}, {}, {}
    for order in range(1, max_petal_dim+1):
        HL[order] = SparseTensor(row=torch.LongTensor([]), col=torch.LongTensor([]), value=torch.tensor([]), sparse_sizes=(n_core, n_core))
        _D[order] = torch.zeros(n_core)
        _delta[order] = torch.ones( (order+1)**2 )

    for cell in get_cells(G, max_petal_dim):
        order = len(cell) - 1
        petalNum[order] += 1
        _D[order][cell] += 1
        comb = torch.LongTensor([[a, b] for a, b in itertools.combinations(cell, 2)])
        rows = torch.cat([torch.LongTensor(cell), comb[:, 0], comb[:, 1]], dim=0)
        cols = torch.cat([torch.LongTensor(cell), comb[:, 1], comb[:, 0]], dim=0)
        HL[order] = HL[order] + SparseTensor(row=rows, col=cols, value=_delta[order], sparse_sizes=(n_core,n_core))

    for order in range(1, max_petal_dim + 1):
        d_inv = _D[order]
        d_inv[d_inv==0] = 1
        d_inv = d_inv.pow_(-0.5)
        HL[order] = mul(HL[order], d_inv.view(1, -1))
        HL[order] = mul(HL[order], d_inv.view(-1, 1)/(order+1))

    return HL, petalNum


def gen_HL_samll_space(G: nx.graph, max_petal_dim=2, petalType='simplex'):
    """
    处理hubOrder=0节点级任务
    这个函数应该已经做到了空间占用最小,但不是很快
    这个函数替换了creat_L_SparseTensor(该函数又调用了creat_L2)
    """
    hubOrder = 0

    def get_simplex(G: nx.graph, max_petal_dim):
        """ 生成所有dim<=max_petal_dim的simplex """
        itClique = nx.enumerate_all_cliques(G)
        nextClique = next(itClique)
        # 跳过所有0团（节点）
        while len(nextClique) <= 1:
            nextClique = next(itClique)
        while len(nextClique) <= max_petal_dim + 1:
            yield sorted(nextClique)
            # 寻找迭代器的下一个团
            try:
                nextClique = next(itClique)
            except StopIteration:
                break

    def get_induced_cycles(G: nx.graph, max_petal_dim):
        """Extracts the induced cycles from a graph using networkx."""
        graph = nx.DiGraph(G)
        max_ring = max_petal_dim + 1

        def is_cycle_edge(i1, i2, cycle):
            if i2 == i1 + 1:
                return True
            if i1 == 0 and i2 == len(cycle) - 1:
                return True
            return False

        def is_chordless(cycle):
            for (i1, v1), (i2, v2) in itertools.combinations(enumerate(cycle), 2):
                if not is_cycle_edge(i1, i2, cycle) and graph.has_edge(v1, v2):
                    return False
            return True

        nx_rings = set()
        for cycle in nx.simple_cycles(graph):
            # Because we need to use a DiGraph for this method, it will also return each edge
            # as a cycle. So we skip these together with cycles above the maximum length.
            if len(cycle) <= 1 or len(cycle) > max_ring:
                continue
            # We skip the cycles with chords
            if not is_chordless(cycle):
                continue

            yield sorted(cycle)

    if petalType == 'simplex':
        get_cells = get_simplex
    elif petalType == 'ring':
        get_cells = get_induced_cycles
    else:
        raise NotImplementedError

    # 数据初始化
    petalNum = np.zeros(max_petal_dim + 1, dtype=int)  # 记录各阶团的数量
    petalNum[0] = n_core = G.number_of_nodes()
    """
    _D:节点所连接的的三角形数
    _delta: 三角形中所包含的节点数，取了倒数
    """
    HL, _D, _delta = {}, {}, {}
    for order in range(1, max_petal_dim + 1):
        #HL[order] = SparseTensor(row=torch.LongTensor([]), col=torch.LongTensor([]), value=torch.tensor([]), sparse_sizes=(n_core, n_core))
        HL[order] = torch.zeros(n_core, n_core)
        _D[order] = torch.zeros(n_core)
        _delta[order] = torch.ones((order + 1) ** 2)

    for cell in get_cells(G, max_petal_dim):
        order = len(cell) - 1
        petalNum[order] += 1
        _D[order][cell] += 1
        comb = torch.LongTensor([[a, b] for a, b in itertools.combinations(cell, 2)])
        rows = torch.cat([torch.LongTensor(cell), comb[:, 0], comb[:, 1]], dim=0)
        cols = torch.cat([torch.LongTensor(cell), comb[:, 1], comb[:, 0]], dim=0)
        #HL[order] = HL[order] + SparseTensor(row=rows, col=cols, value=_delta[order], sparse_sizes=(n_core, n_core))
        HL[order][rows, cols] += 1

    for order in range(1, max_petal_dim + 1):
        d_inv = _D[order]
        d_inv[d_inv == 0] = 1
        d_inv = d_inv.pow_(-0.5)
        HL[order] = SparseTensor.from_dense(HL[order])
        HL[order] = mul(HL[order], d_inv.view(1, -1))
        HL[order] = mul(HL[order], d_inv.view(-1, 1) / (order + 1))

    return HL, petalNum


# def get_induced_cycles(G, max_ring):
#     """Extracts the induced cycles from a graph using networkx."""
#     graph = nx.DiGraph(G)
#     ringNum = np.zeros(max_ring, dtype=int)
#     ringNum[0] += G.number_of_nodes()
#
#     def is_cycle_edge(i1, i2, cycle):
#         if i2 == i1 + 1:
#             return True
#         if i1 == 0 and i2 == len(cycle) - 1:
#             return True
#         return False
#
#     def is_chordless(cycle):
#         for (i1, v1), (i2, v2) in itertools.combinations(enumerate(cycle), 2):
#             if not is_cycle_edge(i1, i2, cycle) and graph.has_edge(v1, v2):
#                 return False
#         return True
#
#     nx_rings = set()
#     for cycle in nx.simple_cycles(graph):
#         # Because we need to use a DiGraph for this method, it will also return each edge
#         # as a cycle. So we skip these together with cycles above the maximum length.
#         if len(cycle) <= 1 or len(cycle) > max_ring:
#             continue
#         # We skip the cycles with chords
#         if not is_chordless(cycle):
#             continue
#         # Store the cycle in a canonical form
#         nx_rings.add(tuple(sorted(cycle)))
#         ringNum[len(cycle)-1] += 1
#
#     return nx_rings, ringNum



def COLLAB():
#if __name__ == '__main__':
    """
    COLLAB数据集太密集，单独为其每个子图跑了一遍程序
    """
    root = 'D:\\PyCharmProject\\cwn\\datasets'
    name = 'COLLAB'
    max_petal_dim = 2
    degree_as_tag={'IMDBBINARY':True, 'IMDBMULTI':True, 'REDDITBINARY':False, 'REDDITMULTI5K':False, 'COLLAB':False,
                   'MUTAG':False, 'PTC':False, 'NCI1':False, 'NCI109':False, 'PROTEINS':False }#下面一行这些数据集不是由label的嘛
    raw_dir = os.path.join(root, name, 'raw')
    load_from = os.path.join(raw_dir, '{}_graph_list_degree_as_tag_{}.pkl'.format(name, degree_as_tag[name]))
    if os.path.isfile(load_from):
        with open(load_from, 'rb') as handle:
            graph_list = pickle.load(handle)

    for idx, graph in enumerate(graph_list):
        print(f"generate HL for graph {idx}({graph.x.shape[0]} nodes and {graph.edge_index.shape[1] / 2} edges)")
        src, dst = graph.edge_index
        edges = zip(src.tolist(), dst.tolist())
        G = nx.from_edgelist(edges)
        G.add_nodes_from(range(graph.x.shape[0]))  # 加这行代码因为可能存在孤立节点

        L, tri_num = creat_L2(G)

        torch.save(L, os.path.join(root, name,'temp_HL', f'{name}_HL_{idx}.pt'))
        with open(os.path.join(root, name,'temp_HL', 'cell_num.txt'), 'a') as file:
            string = f'{G.number_of_nodes()}, {G.number_of_edges()}, {tri_num:.0f}'
            file.write(string + '\n')


    print("HH")

if __name__ == '__main__':
    root = 'D:\\PyCharmProject\\cwn\\datasets'
    name = 'COLLAB'
    simplex_num = np.loadtxt(os.path.join(root, name,'temp_HL', 'cell_num.txt'), delimiter=',')
    n_graph = simplex_num.shape[0]
    with open(os.path.join(root, name, name + '_simplex_statics.yaml'), "w") as f:
        statistics = {'graph_num': n_graph,
                      'n_simplex': np.sum(simplex_num, axis=0).tolist()}
        yaml.dump(statistics, f)
    HL=[]
    for idx in range(n_graph):
        L = torch.load(os.path.join(root, name,'temp_HL', f'{name}_HL_{idx}.pt'))
        HL.append(L)
    torch.save(HL, os.path.join(root, name, name+'_HL.pt'))
