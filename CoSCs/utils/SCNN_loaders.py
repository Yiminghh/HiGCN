
import os
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import random
from torch_geometric.data import Data
def normalize2(L, Lx, half_interval=False):
    # 该函数用于将拉普拉斯矩阵 Lx 根据拉普拉斯矩阵 L 进行归一化处理。
    assert (sp.isspmatrix(L))
    M = L.shape[0]
    assert (M == L.shape[1])
    topeig = spl.eigsh(L, k=1, which="LM", return_eigenvectors=False)[
        0]  # we use the maximal eigenvalue of L to normalize
    # print("Topeig = %f" %(topeig))

    ret = Lx.copy()
    if half_interval:
        ret *= 1.0 / topeig
    else:
        ret *= 2.0 / topeig
        ret.setdiag(ret.diagonal(0) - np.ones(M), 0)

    return ret


def coo2tensor(A):
    assert(sp.isspmatrix_coo(A))
    idxs = torch.LongTensor(np.vstack((A.row, A.col)))
    vals = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(idxs, vals, size = A.shape, requires_grad = False)

def SNNLoader(args):
    name = args.dataset
    root = os.path.join('.','data', name)

    topdim = 0
    laplacians = np.load(os.path.join(root, 'SCNN', 'hodge_laplacians.npy'), allow_pickle=True)
    boundaries = np.load(os.path.join(root, 'SCNN', 'boundaries.npy'), allow_pickle=True)

    # 将拉普拉斯矩阵进行归一化
    Ls = [coo2tensor(normalize2(laplacians[i], laplacians[i], half_interval=True)) for i in range(topdim + 1)]
    Ds = [coo2tensor(boundaries[i].transpose()) for i in range(topdim + 1)]
    adDs = [coo2tensor(boundaries[i]) for i in range(topdim + 1)]

    y = torch.from_numpy(np.loadtxt(os.path.join(root, '0-simplex', 'influence.txt')))
    node_num = len(y)

    # the missing values are replaced by the median of the knowns
    cochain_input_alldegs = torch.mean(y) * torch.ones_like(y)
    random_miss_id = torch.tensor(random.sample(range(node_num), int(node_num * args.train_rate)))
    cochain_input_alldegs[random_miss_id] = y[random_miss_id]
    cochain_input_alldegs = [cochain_input_alldegs.to(torch.float32).view(1,1,-1)]

    train_mask = torch.zeros(node_num, dtype=torch.bool)
    train_mask[random.sample(range(node_num), int(node_num * args.train_rate))] = True

    data = Data(xs=cochain_input_alldegs, Ls = Ls, Ds = Ds, adDs = adDs,
                y=y.view(-1, 1), train_mask=train_mask)

    return data


def SCNNLoader(args):
    name = args.dataset
    root = os.path.join('.','data', name)

    topdim = 0
    laplacians = np.load(os.path.join(root, 'SCNN', 'hodge_laplacians.npy'), allow_pickle=True)
    laplacians_down = np.load(os.path.join(root, 'SCNN', 'laplacians_down.npy'), allow_pickle=True)
    laplacians_up = np.load(os.path.join(root, 'SCNN', 'laplacians_up.npy'), allow_pickle=True)
    boundaries = np.load(os.path.join(root, 'SCNN', 'boundaries.npy'), allow_pickle=True)

    # 将拉普拉斯矩阵进行归一化
    Ls = [coo2tensor(normalize2(laplacians[i], laplacians[i], half_interval=True)) for i in range(topdim + 1)]
    Ds = [coo2tensor(boundaries[i].transpose()) for i in range(topdim + 1)]
    adDs = [coo2tensor(boundaries[i]) for i in range(topdim + 1)]
    Lls = [coo2tensor(normalize2(laplacians[i], laplacians_down[i], half_interval=True)) for i in range(topdim + 1)]
    Lus = [coo2tensor(normalize2(laplacians[i], laplacians_up[i], half_interval=True)) for i in range(topdim + 1)]
    print(torch.max((Lus[0].to_dense()-Ls[0].to_dense())))
    y = torch.from_numpy(np.loadtxt(os.path.join(root, '0-simplex', 'influence.txt')))
    node_num = len(y)

    # the missing values are replaced by the median of the knowns
    cochain_input_alldegs = torch.mean(y) * torch.ones_like(y)
    random_miss_id = torch.tensor(random.sample(range(node_num), int(node_num * args.train_rate)))
    cochain_input_alldegs[random_miss_id] = y[random_miss_id]
    cochain_input_alldegs = [cochain_input_alldegs.to(torch.float32).view(1,1,-1)]

    train_mask = torch.zeros(node_num, dtype=torch.bool)
    train_mask[random.sample(range(node_num), int(node_num * args.train_rate))] = True

    data = Data(xs=cochain_input_alldegs, Lls = Lls, Lus = Lus, Ds = Ds, adDs = adDs,
                y=y.view(-1, 1), train_mask=train_mask)

    return data