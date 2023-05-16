import sys
#sys.path.append("..")
from node_classify.utils.dataset_utils import DataLoader, random_planetoid_splits
from node_classify.utils.param_utils import *
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
            if args.net == 'HiGCN':
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
    args = parse_args()
    Net = get_net(args.net)

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
    print(f'{args.net} on dataset {args.dataset}, in {RPMAX} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')


