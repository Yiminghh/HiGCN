import sys
sys.path.append("..")
import argparse
from utils.dataset_utils import DataLoader
from utils.utils import random_planetoid_splits
from models.HiSCN_model import HiSCN
from models.benchmarks import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.utils import homophily


def RunExp(args, dataset, data, Net, percls_trn, val_lb):
    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data)[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    appnp_net = Net(dataset, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    model, data = appnp_net.to(device), data.to(device)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
                'params': model.lin2.parameters(),
                'weight_decay': args.weight_decay, 'lr': args.lr
            },
            {
                'params': model.prop1.parameters(),
                'weight_decay': 0.0, 'lr': args.lr
            }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history, val_acc_history = [], []
    Gamma_0, Gamma_1 = [], []
    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            # best_model_wts = copy.deepcopy(model.state_dict())
            if args.net == 'HiSCN':
                TEST1 = appnp_net.hgc[0].fW.clone()
                Alpha1 = TEST1.detach().cpu().numpy()
                TEST2 = appnp_net.hgc[1].fW.clone()
                Alpha2 = TEST2.detach().cpu().numpy()
                Gamma_0 = abs(Alpha1)
                Gamma_1 = abs(Alpha2)

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
    return test_acc, best_val_acc, Gamma_0, Gamma_1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Texas_null', help='dataset name')
    parser.add_argument('--RPMAX', type=int, default=100, help='repeat times')
    parser.add_argument('--Order', type=int, default=2,help='max simplix dimension')
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
                        choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN', 'BernNet', 'HiSCN'],
                        default='HiSCN',
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

    """
    这边的参数后面要删掉
    """
    # args.dataset='Texas_null'
    # args.lr=0.1
    # args.alpha=0.5
    # args.weight_decay=0.0001
    # args.dprate=0.7
    # args.dropout=0.7
    # # args.net = 'HiSCN'
    # args.RPMAX = 10
    # args.early_stopping=200

    gnn_name = args.net
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
    elif gnn_name == 'HiSCN':
        Net = HiSCN

    dataset, data = DataLoader(args.dataset, args)

    RPMAX = args.RPMAX
    homo = homophily(data.edge_index, data.y)
    Gamma_0 = None
    alpha = args.alpha
    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(val_rate * len(data.y)))
    TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)
    print('True Label rate: ', TrueLBrate)

    args.C = len(data.y.unique())
    args.Gamma = Gamma_0

    Results0 = []
    Result_test = []
    Result_val = []

    for RP in tqdm(range(RPMAX)):
        test_acc, best_val_acc, Gamma_0, Gamma_1 = RunExp(args, dataset, data, Net, percls_trn, val_lb)

        Results0.append([test_acc, best_val_acc])
        Result_test.append(test_acc)
        Result_val.append(best_val_acc)

        print(f'test_acc:{test_acc:.4f}, best_val_acc:{best_val_acc:.4f}\n')
        with open('./results/log_hand_' + args.dataset + '.txt', 'a') as f:
            f.write(f'test_acc:{test_acc:.4f}, best_val_acc:{best_val_acc:.4f}\n')

    test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
    test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
    print(f'{gnn_name} on dataset {args.dataset}, in {RPMAX} repeated experiment:')
    print(
        f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')


