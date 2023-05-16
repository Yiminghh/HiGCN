#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8


import argparse
from node_classify.models.HiGCN_model import HiGCN
from node_classify.models.benchmarks import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora', help='dataset name')
    parser.add_argument('--RPMAX', type=int, default=100, help='repeat times')
    parser.add_argument('--Order', type=int, default=2, help='max simplix dimension')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--train_rate', type=float, default=0.6)
    parser.add_argument('--val_rate', type=float, default=0.2)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--rho', type=str,
                        choices=['-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0.1', '0.2', '0.3', '0.4', '0.5'],
                        default='0.1', help='adjustable triangle density for null model')
    parser.add_argument('--net', type=str,
                        choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN', 'BernNet', 'HiGCN'],
                        default='HiGCN',
                        )
    """
    The following arguments are used in the benchmarks!
    """
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str, default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', choices=['PPNP', 'GPR_prop'], default='GPR_prop')
    parser.add_argument('--Bern_lr', type=float, default=0.002, help='learning rate for BernNet propagation layer.')
    args = parser.parse_args()

    return args

def get_net(gnn_name):
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'JKNet':
        Net = GCN_JKNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
    elif gnn_name == 'BernNet':
        Net = BernNet
    elif gnn_name == 'HiGCN':
        Net = HiGCN

    return Net