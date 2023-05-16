import torch
import os
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import random

def datanorm(value):
    return (value-torch.mean(value))/(torch.max(value)-torch.min(value))

def coDataLoader(args, printData=True):

    name = args.dataset
    maxOrder = args.Order
    hubOrder = args.hubOrder
    if name not in ['Geology','History','DBLP']:
        raise ValueError(f'dataset {name} not supported in dataloader')
    assert hubOrder == 0, "hubOrder should be zero for node-level tasks!"

    root = os.path.join('.', 'data', name, f'{hubOrder}-simplex')

    HL = torch.load(os.path.join(root, f'HL_{name}.pt'))
    for order in list(HL.keys()):
        if order > args.Order:
            HL.pop(order, None)

    y = torch.from_numpy(np.loadtxt(os.path.join(root, 'influence.txt')))
    node_num = len(y)

    # degree = torch.from_numpy(np.loadtxt(os.path.join(root, 'degree.txt'))).view(-1, 1)
    # ND = torch.from_numpy(np.loadtxt(os.path.join(root, 'ND.txt'))).view(-1, 1)
    # hIndex = torch.from_numpy(np.loadtxt(os.path.join(root, 'Coreness.txt'))).view(-1, 1)
    # Core = torch.from_numpy(np.loadtxt(os.path.join(root, 'hIndex.txt'))).view(-1, 1)

    #the missing values are replaced by the median of the knowns
    signal = torch.mean(y)*torch.ones_like(y)
    random_miss_id = torch.tensor(random.sample(range(node_num), int(node_num*args.train_rate)))
    signal[random_miss_id] = y[random_miss_id]
    signal = signal.to(torch.float32).view(-1, 1)


    train_mask = torch.zeros(node_num, dtype=torch.bool)
    train_mask[random.sample(range(node_num), int(node_num*args.train_rate))] = True

    #data = Data(x=torch.cat((degree, ND, hIndex, Core, signal), dim=1).float(), HL=HL, y=y.view(-1, 1), train_mask=train_mask)
    data = Data(x=signal, HL=HL, y=y.view(-1, 1), train_mask=train_mask)
    if printData:
        print(data)
    print("Finish load data!")

    return data

