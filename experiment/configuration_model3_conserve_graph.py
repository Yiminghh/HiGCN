import networkx as nx
import random
import numpy as np
import copy
import os
import itertools
import time
import pandas as pd

np.random.seed(0)
random.seed(0)


def find_clique(G, k):
    '''
    #团计算方法
    Parameters
    ----------
    G : 网络G.
    k : 阶数，1,2,3,4,5,...

    Returns
    -------
    cliquesk : 团的列表

    '''

    cliques = nx.enumerate_all_cliques(G)
    cliquesk = []
    for clq in cliques:
        if len(clq) < (k + 1):
            continue
        if len(clq) == (k + 1):
            cliquesk.append(clq)
        if len(clq) > (k + 1):
            break
    cliquesk = sorted([tuple(sorted(i)) for i in cliquesk])
    return cliquesk


####该函数用于判断是否无三角形
def judege_triangle(graph, edge):
    neighbor1 = list(nx.neighbors(graph, str(edge[0])))
    neighbor2 = list(nx.neighbors(graph, str(edge[1])))
    interesction = list(set(neighbor1) & set(neighbor2))
    if len(interesction) == 0:
        return True  # 无三角形
    else:
        return False  # 有三角形


####计算应该删除的2-clique
def remove_2_clique(graph, mk, node2, node3, node4, node5):
    clique_r = []
    # 2和3的共同邻居
    interesction_2_3 = list(set(nx.neighbors(graph, node2)) & set(nx.neighbors(graph, node3)))
    # 组成2-clique
    for i in interesction_2_3:
        clique_r.append(tuple(sorted([i, node2, node3])))
    # 4和5的共同邻居
    interesction_4_5 = list(set(nx.neighbors(graph, node4)) & set(nx.neighbors(graph, node5)))
    # 组成2-clique
    for i in interesction_4_5:
        clique_r.append(tuple(sorted([i, node4, node5])))
    mk_new = list(set(mk) - set(clique_r))
    return mk_new


###############该函数用于判断满足条件的三角形
def get_satisfied_edge(graph, Edges, node1, node2, node3):
    node4_suitable = set(graph.nodes())
    node2_neibor = list(nx.neighbors(graph, node2))  # node 4 不能是node2的邻居
    node4_suitable = node4_suitable - set(node2_neibor)
    for i in node2_neibor:
        node4_suitable = node4_suitable - set(nx.neighbors(graph, i))
    if len(node4_suitable) == 0:
        return -1, -1

    node5_suitable = set(graph.nodes())
    node3_neibor = list(nx.neighbors(graph, node3))
    node5_suitable = node5_suitable - set(node3_neibor)
    for i in node3_neibor:
        node5_suitable = node5_suitable - set(nx.neighbors(graph, i))
    if len(node5_suitable) == 0:
        return -1, -1
    random.shuffle(Edges)
    for e in Edges:
        if (e[0] in node4_suitable) & (e[1] in node5_suitable):
            return e[0], e[1]
    return -1, -1


####该函数用于选择合适的节点
def get_suitable_node(graph, repick, max_repick):
    while True:
        if repick > max_repick:
            return -1, -1, -1, -1, -1, repick
        repick = repick + 1
        print(repick)
        node1 = random.choice(list(graph.nodes()))
        if graph.degree(node1) < 2:  # 度>2
            continue
        # 选择度>=2的邻居
        neighbors = [i for i in list(nx.neighbors(graph, node1)) if graph.degree(i) >= 2]
        # neighbors = [i for i in list(nx.neighbors(graph, str(node1))) if graph.degree(i) >= 2]
        if len(neighbors) < 2:
            continue
        # 保证2和3无连边
        combine_2_3 = []
        for i in list(itertools.combinations(neighbors, 2)):
            if graph.has_edge(i[0], i[1]) == False:
                combine_2_3.append(i)
        if len(combine_2_3) < 1:
            continue
        random.shuffle(combine_2_3)  # 打乱后随机选择
        node4, node5 = -1, -1
        for i in range(len(combine_2_3)):
            node2, node3 = combine_2_3[i][0], combine_2_3[i][1]  # node2，node3

            node2_neighbor = []
            # for i in list(nx.neighbors(graph, str(node2))):
            for i in list(nx.neighbors(graph, node2)):
                # if not (set(nx.neighbors(graph, str(node2))) & set(nx.neighbors(graph, i))):
                if not (set(nx.neighbors(graph, node2)) & set(nx.neighbors(graph, i))):
                    if i != node1:
                        node2_neighbor.append(i)

            node3_neighbor = []
            for i in list(nx.neighbors(graph, node3)):  # str(node3)
                if not (set(nx.neighbors(graph, node3)) & set(nx.neighbors(graph, i))): # str(node3)
                    if i != node1:
                        node3_neighbor.append(i)

            # 保证node4和node5无连边
            combine_4_5 = []
            for i in node2_neighbor:
                for j in node3_neighbor:
                    if (graph.has_edge(i, j) == False) & (i != j):
                        combine_4_5.append((i, j))
            if len(combine_4_5) < 1:
                continue
            node_4_5 = random.choice(combine_4_5)
            node4, node5 = node_4_5[0], node_4_5[1]  # node4，node5
            break
        if (node4 == -1) | (node5 == -1):
            continue
        # return node1, str(node2), str(node3), str(node4), str(node5), repick
        return node1, node2, node3, node4, node5, repick


def get_remove_triangle(graph, mk, repick, max_repick):
    Edges = list(graph.edges())
    while True:
        if repick > max_repick:
            return -1, -1, -1, -1, -1, repick
        if len(mk) == 0:
            repick = max_repick + 1
            continue
        triangle = list(random.choice(mk))  # （1）
        random.shuffle(triangle)
        repick = repick + 1  # 重新选择边的次数
        combine_2_3 = list(itertools.combinations(triangle, 2))
        # node1, node2, node3, node4, node5 = -1, -1, -1, -1, -1
        for edge in combine_2_3:
            node2, node3 = edge[0], edge[1]
            node1 = list(set(triangle) - set(edge))[0]
            node4, node5 = get_satisfied_edge(graph, Edges, node1, node2, node3)
            if (node4 == -1) | (node5 == -1):
                continue
            return node1, node2, node3, node4, node5, repick


###保存txt文件
def save_graph(graph, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    graph_out = open(filename, 'w')
    edges = graph.edges()
    for j in edges:
        if j[0] != j[1]:
            graph_out.write(str(j[0]) + " " + str(j[1]) + "\n")
            # graph_out.write(str(j[1]) + " " + str(j[0]) + "\n")
    graph_out.close()


# 增加或减少三角形的数量
def generate_or_remove_triangles(graph, iteration_time, p1, filename, k, origin_triangle, add_ratio, max_repick=1000):
    jj = 0  # 迭代轮数
    count = 0
    repick = 0  # 重新选择的次数
    if p1 == 1:
        mk = find_clique(graph, 2)
    # while jj < iteration_time[-1]:
    Tag = True
    while Tag:
        if repick > max_repick:
            print('quit,iterations is ' + str(jj) + ',repick num is', str(repick))
            print(jj, end=' ')
            triangle = nx.triangles(graph)
            print("高阶结构数量为:", sum(triangle.values()) / 3)
            # 文件保存路径 path
            path = "results/" + filename + "/" + str(k) + "/" + filename + "_" + str(count + 1) + "_" + positive_symbol[
                p1] + ".txt"
            save_graph(graph, path)
            break
        if p1 == 0:  # 增加三角形
            node1, node2, node3, node4, node5, repick = get_suitable_node(graph, repick, max_repick)

            if (node1 == -1) | (node2 == -1) | (node3 == -1) | (node4 == -1) | (node5 == -1):
                continue
            graph.remove_edge(node2, node4)
            graph.remove_edge(node3, node5)
            graph.add_edge(node2, node3)
            graph.add_edge(node4, node5)
            if nx.is_connected(graph):  # 下一轮迭代
                jj = jj + 1
                repick = 0  # 重新选择的次数
                new_triangle = sum(nx.triangles(graph).values()) / 3
                if new_triangle / origin_triangle >= 1 + add_ratio:
                    Tag = False  # 跳出循环
            else:  # 恢复
                graph.add_edge(node2, node4)
                graph.add_edge(node3, node5)
                graph.remove_edge(node2, node3)
                graph.remove_edge(node4, node5)

        if p1 == 1:  # 减少三角形
            node1, node2, node3, node4, node5, repick = get_remove_triangle(graph, mk, repick, max_repick)
            if (node1 == -1) | (node2 == -1) | (node3 == -1) | (node4 == -1) | (node5 == -1):
                continue
            graph.remove_edge(node2, node3)
            graph.remove_edge(node4, node5)
            graph.add_edge(node2, node4)
            graph.add_edge(node3, node5)
            if nx.is_connected(graph):
                jj = jj + 1
                repick = 0  # 重新选择的次数
                # 删除一些三角形
                mk = remove_2_clique(graph, mk, node2, node3, node4, node5)
                new_triangle = sum(nx.triangles(graph).values()) / 3
                if new_triangle / origin_triangle <= 1 + add_ratio:
                    Tag = False  # 跳出循环
            else:
                graph.add_edge(node2, node3)
                graph.add_edge(node4, node5)
                graph.remove_edge(node2, node4)
                graph.remove_edge(node3, node5)
        # if (jj == iteration_time[count]):
    print(jj, end=' ')
    triangle = nx.triangles(graph)
    print("原有的高阶结构数量为:", origin_triangle)
    print("现有的高阶结构数量为:", sum(triangle.values()) / 3)
    # 文件保存路径 path
    path = "results/" + filename + "/" + str(k) + "/" + filename + "_" + str(count + 1) + "_" + positive_symbol[
        p1] + str(add_ratio)+ '2' + ".txt"
    save_graph(graph, path)
    count = count + 1


def Simplicial_Null_Model(dataset, p1, file, add_ratio, interation_time, repeat):
    ###第一步，网络读取
    lines = pd.read_csv(dataset)
    ggg = nx.DiGraph()
    ggg.add_edge(int(lines.keys()[0].split('\t')[0]), int(lines.keys()[0].split('\t')[1]))
    for line in lines.values:
        ggg.add_edge(int(line[0].split('\t')[0]), int(line[0].split('\t')[1]))
    wuyu = []
    for edge in ggg.edges():
        if edge[::-1] in ggg.edges():  # check if reverse edge exist in graph
            wuyu.append(edge[::-1])
    print(len(wuyu))

    G = nx.read_weighted_edgelist(dataset, create_using=nx.Graph)
    # G = nx.barabasi_albert_graph(1000, 3)
    if p1 == 1:
        triangle = nx.triangles(G)
        max_r = 1 * (sum(triangle.values()) / 3)  # 最大重选次数
    else:
        max_r = 1 * nx.number_of_nodes(G)  # 最大重选次数为节点数量
    # 增加或减少高阶结构
    # for i in range(1, repeat + 1):  # 重复10次，保存10个网络
    #     print(i)
    copy_graph = copy.deepcopy(G)
    origin_triangle = sum(nx.triangles(G).values()) / 3
    generate_or_remove_triangles(copy_graph, list(iteration_time), p1, file, k=1, add_ratio=add_ratio, origin_triangle=origin_triangle,
                                     max_repick=max_r)


if __name__ == '__main__':
    positive_symbol = ['generate', 'remove']
    start = time.time()
    network_name = 'Texas'  # 网络名称
    print(network_name)
    iteration_time = [1]
    # iteration_time = np.linspace(10, 100, 10, dtype=int)  # 10-100轮，每10轮保存一次
    print(iteration_time)
    # dataset_path = 'datasets/' + network_name + '.txt'
    dataset_path = '..\\data\\texas\\texas_edge_list.txt'
    # dataset_path = 'D:\PycharmProjects\pythonProject\HiSCN\data\cSBM_1000\csbm_1000.txt'
    for add_ratio in [0.1]:
    # for add_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
    # dataset_path-----原始网络数据路径
    # 0表示增加高阶结构，1表示减少高阶结构
    # interation_time---保存轮数
    # repeat----重复次数
    # path(generate_or_remove_triangles函数中)-----配置模型保存路径
        Simplicial_Null_Model(dataset_path, 0, network_name, add_ratio=add_ratio, interation_time=iteration_time, repeat=1)
    # end = time.time()
    # print(end - start)
