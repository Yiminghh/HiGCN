import os
import numpy as np
import itertools
import yaml


if __name__ == '__main__':
    name = 'History'
    maxOrder = 5
    tri_threshold = 3  # 大于tri_threshold的才视作单纯形，其余的为开三角形

    assert name in ['DBLP', 'History', 'Geology']

    if name == 'DBLP':
        simplex_size_list = np.loadtxt(f'./{name}/raw/coauth-DBLP-nverts.txt', dtype=int)
        simplex_node_list = np.loadtxt(f'./{name}/raw/coauth-DBLP-simplices.txt', dtype=int)
        simplex_time_label = np.loadtxt(f'./{name}/raw/coauth-DBLP-times.txt', dtype=int)
        with open(f'./{name}/raw/coauth-DBLP-node-labels.txt', 'r') as f:
            lines = f.readlines()
    elif name in ['History', 'Geology']:
        simplex_size_list = np.loadtxt(f'./{name}/raw/coauth-MAG-{name}-nverts.txt', dtype=int)
        simplex_node_list = np.loadtxt(f'./{name}/raw/coauth-MAG-{name}-simplices.txt', dtype=int)
        simplex_time_label = np.loadtxt(f'./{name}/raw/coauth-MAG-{name}-times.txt', dtype=int)
        with open(f'./{name}/raw/coauth-MAG-{name}-node-labels.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

    if name == 'DBLP':
        start_year = 2017
    elif name == 'History':
        start_year = 2013
    elif name == 'Geology':
        start_year = 2015


    node_label = [line.strip().split(' ', 1)[1] for line in lines]





    n_ = np.zeros(maxOrder + 1, dtype=int) #记录各阶单纯形数量
    SCs = {order:{} for order in range(26)}
    node_index = 0  # 目前读到的地方
    for idx, simplex_size in enumerate(simplex_size_list):
        print(idx, node_index)
        temp_simplex = []
        if simplex_time_label[idx] >= start_year:
            for temp_node_id in range(simplex_size):
                temp_simplex.append(simplex_node_list[node_index+temp_node_id])
        node_index += simplex_size
        temp_simplex = set(temp_simplex)#很奇怪DBLP数据中会出现[67554, 10096, 19553, 291402, 287130, 99739, 10096]会有重复值

        for r in range(1, min(len(temp_simplex)+1, maxOrder+2)):

            for sub in itertools.combinations(temp_simplex, r):
                order = len(sub) - 1
                simplex = tuple(sorted(sub))
                if simplex not in SCs[order]:
                    SCs[order][simplex] = 1
                else:
                    SCs[order][simplex] += 1

    print("writing file!")
    node_dict = {} #原始节点:新节点编号
    new_id = 0
    for node in sorted(SCs[0].keys()):
        if SCs[0][(node)] >= tri_threshold:
            node_dict[node[0]] = new_id
            new_id += 1

    for k in range(0, maxOrder+1):
        path = f'./{name}/{k}-simplex'
        os.makedirs(path, exist_ok=True)

        sorted_keys = sorted(SCs[k].keys())
        sorted_keys = [elem for elem in sorted_keys if SCs[k][elem] >= tri_threshold]
        value_list = [SCs[k][elem] for elem in sorted_keys]
        n_[k] = len(value_list)
        with open(f'./{name}/{k}-simplex/simplex-index.txt', 'w') as f:
            for tup in sorted_keys:
                f.write(' '.join(str(  node_dict[old_idx]  ) for old_idx in tup) + '\n')
        with open(f'./{name}/{k}-simplex/influence.txt', 'w') as f:
            for item in value_list:
                f.write(str(item) + '\n')

    print("file size:{}, {}, {}".format(simplex_size_list.shape, simplex_node_list.shape, simplex_time_label.shape))
    print(f"year:{min(simplex_time_label)} ~ {max(simplex_time_label)}")

    # 记录各阶单纯形的数量
    with open(os.path.join('.', name, name + '_statistics.yaml'), "w") as f:
        statistics = {'n_simplex': n_.tolist()}
        yaml.dump(statistics, f)

    # 记录节点的映射关系
    with open(f'./{name}/{name}-node-labels.txt', 'w', encoding='utf-8') as f:
        for old_idx in node_dict.keys():
            f.write(f'{node_dict[old_idx]}, {old_idx}, {node_label[old_idx-1]}\n')

    print("Finish!")


