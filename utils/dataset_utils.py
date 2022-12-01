#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch

import os
import os.path as osp
import pickle
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor
import pandas as pd
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import homophily
from utils.gen_HoHLaplacian import creat_L_SparseTensor
import torch
import numpy as np
import networkx as nx
from torch_sparse import SparseTensor, fill_diag, matmul, mul


class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):

        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class WebKB(InMemoryDataset):
    r"""The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def DataLoader(name, args):
    # calculate higher_order adj-matrix and  save
    calculate_ = True if args.net in ['HiSCN','HIMnet'] else False
    hl_path = osp.join('..\\data\\' + name + '\\HL_' + name + '.pt') # 存储hl的路径

    if name in ['cora', 'citeseer', 'pubmed']:
        root_path = '../'
        path = osp.join(root_path, 'data', name)
        dataset = Planetoid(path, name=name)
        data = dataset[0]
    elif name in ['computers', 'photo']:
        root_path = '../'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, T.NormalizeFeatures())
        data = dataset[0]
    elif name in ['chameleon', 'squirrel']:
        # use everything from "geom_gcn_preprocess=False" and
        # only the node label y from "geom_gcn_preprocess=True"
        preProcDs = WikipediaNetwork(
            root='../data/', name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root='../data/', name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
        # return dataset, data
    elif name in ['film']:
        dataset = Actor(
            root='../data/film', transform=T.NormalizeFeatures())
        data = dataset[0]
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root='../data/',
                        name=name, transform=T.NormalizeFeatures())

        data = dataset[0]
    elif name in ['Texas_null']:
        """
        Texas_null is a null model to test different effect of higher-order structures
        """
        name = 'Texas'
        path = '../data/nullModel_Texas/'
        dataset = WebKB(root='../data/',
                        name=name, transform=T.NormalizeFeatures())
        data = dataset[0]
        change = args.rho

        G = nx.Graph()
        graph_edge_list = []
        #dataset_path = '..\\data\\nullModel_' + name + '\\' + name + '_1_generate' + change + '.txt'
        dataset_path = osp.join('..','data','nullModel_'+name, name+'_1_generate' + change + '.txt')
        lines = pd.read_csv(dataset_path)
        G.add_edge(int(lines.keys()[0].split(' ')[0]), int(lines.keys()[0].split(' ')[1]))
        graph_edge_list.append([int(lines.keys()[0].split(' ')[0]), int(lines.keys()[0].split(' ')[1])])
        graph_edge_list.append([int(lines.keys()[0].split(' ')[1]), int(lines.keys()[0].split(' ')[0])])
        for line in lines.values:
            G.add_edge(int(line[0].split(' ')[0]), int(line[0].split(' ')[1]))
            graph_edge_list.append([int(line[0].split(' ')[0]), int(line[0].split(' ')[1])])
            graph_edge_list.append([int(line[0].split(' ')[1]), int(line[0].split(' ')[0])])

        data.edge_index = torch.tensor(graph_edge_list).H
        # data.HL = creat_L_SparseTensor(new_graph, maxCliqueSize=args.Order)
        #calculate_ = False
        hl_path = osp.join('..', 'data', 'nullModel_' + name, name + '_1_generate' + change + '_HL.pt')# 存储hl的路径

    else:
        raise ValueError(f'dataset {name} not supported in dataloader')




    if calculate_ and osp.exists(hl_path):
        data.HL = torch.load(hl_path)
        calculate_ = False
        if len(data.HL) < args.Order:
            calculate_ = True

    if calculate_:
        try:
            # G has been defined in the null model
            print("Runing Null model", G.number_of_nodes())
        except:
            G = nx.Graph()
            G.add_nodes_from(range(data.num_nodes))
            G.add_edges_from(data.edge_index.numpy().transpose())

        print("Calucating higher-order laplacian matix...")
        print(hl_path)
        data.HL = creat_L_SparseTensor(G, maxCliqueSize=args.Order)
        torch.save(data.HL, hl_path)

    homo = homophily(data.edge_index, data.y)
    print("Home:", homo)
    print("Finish load data!")
    return dataset, data


class MyTransform(object):
    def __call__(self, data):
        data.y = data.y[:, 8]  # Specify target: 0 = mu
        return data


class MyFilter(object):
    def __call__(self, data):
        return not (data.num_nodes == 7 and data.num_edges == 12) and \
               data.num_nodes < 450
