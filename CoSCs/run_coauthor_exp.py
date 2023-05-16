import copy
import sys

import scipy.stats

#sys.path.append(".")
#import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import argparse
from utils.coDataLoader import coDataLoader
from utils.SCNN_loaders import SNNLoader, SCNNLoader
from models.co_models import co_HiGCN
from models.SCNNs import SNN, SCNN
import os
import gc
import itertools
import torch.nn as nn


def RunExp(args, data, model, print_log=False):
    '''
    可选择的loss有 = nn.L1Loss()，nn.MSELoss()
    :param args:
    :param data: 构建的Data
    :param Net: 输入的网络模型
    :return:
    '''


    def train(data, model, criterion, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]

        loss = criterion(data.y[data.train_mask], out)

        # optimizer.zero_grad()  # 清空过往梯度，在有batch的时候用
        loss.backward()
        optimizer.step()
        del out
        return loss.detach().cpu()

    @torch.no_grad()
    def test(data, model, criterion):
        model.eval()
        pred = model(data).detach().cpu().numpy()
        y = data.y.detach().cpu().numpy()
        corr = scipy.stats.kendalltau(pred,y)[0]
        return corr


    criterion = nn.L1Loss(reduction="mean")#sum
    #criterion = new_rank_loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_corr = -1.0
    val_loss_history = []
    train_log, acc_log = [], []


    for epoch in range(args.epochs):
        train_loss = train(data, model, criterion, optimizer)

        full_corr = test(data, model, criterion)

        if print_log:
            print("epoch:{} train_Loss:{}, full_corr:{}".format(epoch, train_loss, full_corr))
            train_log.append([epoch, train_loss])
            acc_log.append([epoch, full_corr])


        if full_corr >= best_corr:  # val_loss < best_val_loss:
            best_corr = full_corr
            best_model_wts = copy.deepcopy(model.state_dict())


    # torch.save(model, path)  在这里得到模型，其实可以进行比较了  下面测试
    # 画图
    # if print_log:
    #     plot_train_val(train_log, val_log, args)


    model.load_state_dict(best_model_wts)
    return best_corr, model



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='History', help='dataset name')
    parser.add_argument('--RPMAX', type=int, default=10, help='repeat times')
    parser.add_argument('--Order', type=int, default=3, help='max simplix dimension')
    #parser.add_argument('--miss_percentage', type=int, default=10, help='percentage_missing_values')
    parser.add_argument('--epochs', type=int, default=300)#1000
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dprate', type=float, default=0.00)
    parser.add_argument('--dropout', type=float, default=0.00)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--hubOrder', type=int, default=0, help='fix to 0')
    parser.add_argument('--train_rate', type=float, default=0.1)#0.1
    parser.add_argument('--net', type=str,  choices=['HiGCN', 'SNN', 'SCNN'], default='HiGCN',)
    args = parser.parse_args()

    return args


def set_args():
    args = parse_args()
    args.dataset = 'Geology'
    args.alpha = 1.0
    args.lr = 0.001
    args.weight_decay = 0.0005
    args.hidden = 32
    args.train_rate = 0.90
    args.Order = 3
    args.net = 'HiGCN'

if __name__ == '__main__':

    args = parse_args()
    #args = set_args()
    print("args = ", args)




    if args.net == 'HiGCN':
        data = coDataLoader(args)
        model = co_HiGCN(data, args)
    elif args.net == 'SNN':
        data = SNNLoader(args)
        model = SNN(data, args)
    elif args.net == 'SCNN':
        data = SCNNLoader(args)
        model = SCNN(data, args)


    best_corr, best_model = RunExp(args, data, model, print_log=True)

    # 测试 best_model
    print("============== values间的相关性 ==============")

    sir_list_true = data.y.cpu().numpy()
    sir_list_pred = best_model(data).detach().cpu().numpy()

    train_mask = data.train_mask.detach().cpu().numpy()
    test_tau, _ = scipy.stats.kendalltau(sir_list_true[train_mask], sir_list_pred[train_mask])
    full_tau, _ = scipy.stats.kendalltau(sir_list_true, sir_list_pred)

    # plt.scatter(sir_list_true, sir_list_pred)
    # plt.show()

    print(f"{args.net}, {args.dataset} miss {args.train_rate} values")
    print(f"train_tau:{test_tau}, full_tau:{full_tau}")