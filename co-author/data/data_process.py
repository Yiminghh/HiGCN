import os
import numpy as np
import copy
import torch
import os
from utils.gen_HoHLaplacian import gen_HL_from_SC_list

def hIndex(neighbor_degree_list) -> int:
    """
    neighbor_degree_list: 单个节点的邻居度的list
    """
    neighbor_degree_list.sort()
    result = 0
    cir_num=len(neighbor_degree_list)
    for i in range(0, cir_num):
        if neighbor_degree_list[i] >= cir_num-i:
            result=cir_num-i
            break

    return result


def core_number(degree_list, nbrs):
    """
    Returns the core number for each vertex.
    degree_list:
    nbrs: list{v: v's neighbours}
    """
    nbrs = copy.deepcopy(nbrs)
    if isinstance(degree_list, list):
        n = len(degree_list)
    elif isinstance(degree_list, np.ndarray) or isinstance(degree_list, torch.Tensor):
        n = degree_list.shape[0]

    degrees = dict( zip(range(n), degree_list) )
    # Sort nodes by degree.
    nodes = sorted(degrees, key=degrees.get)
    bin_boundaries = [0]
    curr_degree = 0
    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}
    # The initial guess for the core number of a node is its degree.
    core = degrees
    #nbrs = {v: list(nx.all_neighbors(G, v)) for v in G}
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    return [core[v] for v in range(n)]



if __name__ == '__main__':
    name = 'History'
    maxOrder = 3
    hubOrder = 0
    assert name in ['DBLP', 'History', 'Geology']

    edges =  np.loadtxt(os.path.join('.', name, '1-simplex','simplex-index.txt'), dtype=int).tolist()
    nodes =  np.loadtxt(os.path.join('.', name, '0-simplex','simplex-index.txt'), dtype=int).tolist()
    n0 = len(nodes)

    degree = np.zeros(n0, dtype=int)
    nbrs = {v: [] for v in range(n0)}
    for a,b in edges:
        degree[a] += 1
        degree[b] += 1
        nbrs[a].append(b)
        nbrs[b].append(a)

    # 计算节点层面的coreness, nd, hIndex指标
    coreness = np.array(core_number(degree, nbrs))
    nd, h_Index = np.zeros(n0), np.zeros(n0)
    for node in range(0, n0):
        nd[node] = sum(degree[nbrs[node]])
        h_Index[node] = hIndex(degree[nbrs[node]])

    root = os.path.join('.', name, '0-simplex')
    # 把0阶的属性写入文件
    np.savetxt(os.path.join(root, 'degree.txt'), degree)
    np.savetxt(os.path.join(root, 'ND.txt'), nd)
    np.savetxt(os.path.join(root, 'Coreness.txt'), coreness)
    np.savetxt(os.path.join(root, 'hIndex.txt'), h_Index)


    for k_ in range(1, maxOrder+1):
        # 对于维度大于1的单纯形取均值
        print("calucate index in order={}".format(k_))
        simplex_list = np.loadtxt(os.path.join('.', name, f'{k_}-simplex', 'simplex-index.txt'), dtype=int).tolist()

        n_simplex = len(simplex_list)
        mean_degree, mean_ND, mean_coreness, mean_hIndex = np.zeros(n_simplex), np.zeros(n_simplex),  np.zeros(n_simplex),  np.zeros(n_simplex)
        for idx in range(n_simplex):
            mean_degree[idx] = np.mean(degree[simplex_list[idx]])
            mean_ND[idx] = np.mean(nd[simplex_list[idx]])
            mean_coreness[idx] = np.mean(coreness[simplex_list[idx]])
            mean_hIndex[idx] = np.mean(h_Index[simplex_list[idx]])

        root = os.path.join('.', name, f'{k_}-simplex')
        np.savetxt(os.path.join(root, 'degree.txt'), mean_degree)
        np.savetxt(os.path.join(root, 'ND.txt'), mean_ND)
        np.savetxt(os.path.join(root, 'Coreness.txt'), mean_coreness)
        np.savetxt(os.path.join(root, 'hIndex.txt'), mean_hIndex)


    print("Finish calucate fatures  ({},  maxOrder={})".format(name,  maxOrder))


    print("Calucating HL")
    HL = {}
    for order in range(1, maxOrder+1):
        simplex_list = np.loadtxt(os.path.join('.', name, f'{order}-simplex','simplex-index.txt'), dtype=int).tolist()
        HL[order] = gen_HL_from_SC_list(number_of_nodes=n0, simplex_list=simplex_list)

    torch.save(HL, os.path.join(os.path.join('.', name, '0-simplex', f'HL_{name}.pt')))
    print("Finish HL")


