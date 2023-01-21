import argparse
import sys
sys.path.append("..")
import scipy.stats as st
import argparse
from utils.dataset_utils import DataLoader, random_planetoid_splits
from utils.param_utils import *
from models.benchmarks import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
from torch_geometric.utils import homophily

import torch
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt

def model_train(model, optimizer, data, dprate, epoch):
    model.train()
    optimizer.zero_grad()
    out = model(data)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()

    optimizer.step()

    epoch_loss = loss / out.shape[0]
    epoch_acc = torch.sum(out.max(1)[1] == data.y[data.train_mask])/out.shape[0]
    # if epoch % args.print_freq == 0:
    #     print(f'epoch: {epoch}, train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc

@torch.no_grad()
def model_test(model, data):
    model.eval()
    logits = model(data)
    accs, losses, preds = [], [], []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        loss = F.nll_loss(model(data)[mask], data.y[mask])

        accs.append(acc)
        preds.append(pred.detach().cpu())
        losses.append(loss.detach().cpu())
    return accs, preds, losses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cornell')  # 更改数据集
    parser.add_argument('--RPMAX', type=int, default=1)  # 重复执行次数,10
    parser.add_argument('--epochs', type=int, default=1000)#1000
    parser.add_argument('--early_stopping', type=int, default=200)  # 200

    parser.add_argument('--Order', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.002)            #0.002
    parser.add_argument('--alpha', type=float, default=0.1)          #0.1
    parser.add_argument('--weight_decay', type=float, default=0.005)#0.0005
    parser.add_argument('--dprate', type=float, default=0.5)         #0.5
    parser.add_argument('--dropout', type=float, default=0.5)        #0.5

    parser.add_argument('--K', type=int, default=10)                 #10
    parser.add_argument('--train_rate', type=float, default=0.6)     #0.025
    parser.add_argument('--val_rate', type=float, default=0.2)       #0.025


    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str, choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', choices=['PPNP', 'GPR_prop'], default='GPR_prop')
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN','HiGCN'],
                        default='HiGCN')

    args = parser.parse_args()

    args.dataset = 'squirrel'
    args.RPMAX = 50
    args.lr = 0.5
    args.alpha = 0.1
    args.weight_decay = 0.000
    args.dprate = 0.6

    dataset, data = DataLoader(args.dataset, args)

    gnn_name = args.net
    Net = get_net(args.net)


    percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(args.val_rate * len(data.y)))
    TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)
    print('True Label rate: ', TrueLBrate)

    args.C = len(data.y.unique())
    args.Gamma = None

    #test_acc, best_val_acc, Gamma_0 = RunExp(args, dataset, data, Net, percls_trn, val_lb)

    appnp_net = Net(dataset, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    model, data = appnp_net.to(device), data.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)


    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    for epoch in range(args.epochs):
        epoch_loss, epoch_acc = model_train(model, optimizer, data, args.dprate, epoch)

        [tmp_train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = model_test(model, data)

        if epoch % args.print_freq == 0:
            print(f'***epoch: {epoch}, train Loss: {train_loss:.4f}, train_acc: {tmp_train_acc:.4f}')
        # 这边确实用到验证集了，保留验证集合最好的数据
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            print("test_val_middle:",test_acc)
            train_acc = tmp_train_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            if args.net in ['GPRGNN'] :
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            elif args.net in ['HeGCN_exp']:
                TEST1 = appnp_net.hgc[0].fW.clone()
                Alpha1 = TEST1.detach().cpu().numpy()
                TEST2 = appnp_net.hgc[1].fW.clone()
                Alpha2 = TEST2.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            #Gamma_0 = Alpha

        # 提前停止
        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break


    model.load_state_dict(best_model_wts)
    print("train_acc:", train_acc)
    print("best_val_acc:", best_val_acc, ",best_val_loss", best_val_loss)
    print("test_acc:", test_acc)
    print(train_acc, best_val_acc, test_acc)
    print("finish")

    # plt.plot(list(range(11)), Alpha1, 'b')
    # plt.plot(list(range(11)), Alpha2, 'y')
    # plt.show()