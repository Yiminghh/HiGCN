import networkx as nx
from utils.dataset_utils import graphLoader
import numpy as np
from scipy.sparse import coo_matrix
import os

def build_boundaries(simplices):
    """Build the boundary operators from a list of simplices.

    Parameters
    ----------
    simplices: list of dictionaries
        List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are sets (of size d
        + 1) of the 0-simplices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.

    Returns
    -------
    boundaries: list of sparse matrices
       List of boundary operators, one per dimension: i-th boundary is in i-th position
    """
    boundaries = list()
    for d in range(1, len(simplices)):
        idx_simplices, idx_faces, values = [], [], []
        for simplex, idx_simplex in simplices[d].items():
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex)
                values.append((-1)**i)
                face = simplex.difference({left_out})
                idx_faces.append(simplices[d-1][face])
        assert len(values) == (d+1) * len(simplices[d])
        boundary = coo_matrix((values, (idx_faces, idx_simplices)),
                                     dtype=np.float32,
                                     shape=(len(simplices[d-1]), len(simplices[d])))
        boundaries.append(boundary)
    return boundaries


def build_laplacians(boundaries):
    """Build the Laplacian operators from the boundary operators.

    Parameters
    ----------
    boundaries: list of sparse matrices
       List of boundary operators, one per dimension.

    Returns
    -------
    laplacians: list of sparse matrices
       List of Laplacian operators, one per dimension: laplacian of degree i is in the i-th position
    """
    laplacians = list()
    laplacians_down = list()
    laplacians_up = list()
    up = coo_matrix(boundaries[0] @ boundaries[0].T)
    laplacians.append(up)
    laplacians_up.append(up)
    laplacians_down.append(coo_matrix(up.shape))
    for d in range(len(boundaries)-1):
        down = boundaries[d].T @ boundaries[d]
        up = boundaries[d+1] @ boundaries[d+1].T
        laplacians.append(coo_matrix(down + up))
        laplacians_up.append(coo_matrix(up))
        laplacians_down.append(coo_matrix(down))
    down = boundaries[-1].T @ boundaries[-1]
    laplacians.append(coo_matrix(down))
    laplacians_down.append(coo_matrix(down))
    return laplacians, laplacians_down, laplacians_up


def get_simplex(G: nx.graph, max_petal_dim):
    """ 生成所有dim<=max_petal_dim的simplex """
    itClique = nx.enumerate_all_cliques(G)
    nextClique = next(itClique)
    while len(nextClique) <= max_petal_dim + 1:
        yield sorted(nextClique)
        # 寻找迭代器的下一个团
        try:
            nextClique = next(itClique)
        except StopIteration:
            break


if __name__ == '__main__':

    name = 'Geology'
    max_simplex_dim = 1
    SCs = {k: {} for k in range(max_simplex_dim + 1)}

    if name in ['cora', 'citeseer', 'pubmed', 'computers', 'photo',
                'chameleon', 'squirrel', 'film', 'texas', 'cornell', 'wisconsin']:
        data, _ = graphLoader(name)
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        G.add_edges_from(data.edge_index.numpy().transpose())

        idx = 0
        for simplex in get_simplex(G, max_petal_dim=max_simplex_dim):
            print(simplex)
            order = len(simplex) - 1
            if len(SCs[order]) <= 0:
                idx = 0
            SCs[order][frozenset(simplex)]=idx
            idx = idx + 1
    elif name in ['History', 'Geology', 'DBLP']:
        node_list = np.loadtxt(os.path.join('.', name, '0-simplex', 'simplex-index.txt'), dtype=int)
        for idx, node in enumerate(node_list): #单独处理节点情形，因为frozenset需要可迭代的对象
            SCs[0][frozenset([node])] = idx
        for k_ in range(1, max_simplex_dim+1):
            simplex_list = np.loadtxt(os.path.join('.', name, f'{k_}-simplex', 'simplex-index.txt'), dtype=int)
            for idx, simplex in enumerate(simplex_list):
                SCs[k_][frozenset(simplex)] = idx


    boundaries = build_boundaries(SCs)
    laplacians, laplacians_down, laplacians_up = build_laplacians(boundaries)

    root = f'../data/{name}/SCNN/'
    os.makedirs(root, exist_ok=True)
    np.save(os.path.join(root, 'hodge_laplacians.npy'), laplacians)
    np.save(os.path.join(root, 'laplacians_down.npy'), laplacians_down)
    np.save(os.path.join(root, 'laplacians_up.npy'), laplacians_up)
    np.save(os.path.join(root, 'boundaries.npy'),  boundaries)
    print(f"Finish process {name}")



