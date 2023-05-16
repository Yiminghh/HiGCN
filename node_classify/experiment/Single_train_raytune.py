import argparse

from ray.tune.search.bayesopt import BayesOptSearch

from node_classify.utils.dataset_utils import DataLoader,  random_planetoid_splits
from node_classify.utils.param_utils import *


import torch
import torch.nn.functional as F
import copy
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune import CLIReporter
import torch.nn as nn
from functools import partial
import os


def model_train(model, optimizer, data, dprate, epoch):
    model.train()
    optimizer.zero_grad()
    out = model(data)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()

    optimizer.step()
    epoch_loss = loss / out.shape[0]
    epoch_acc = torch.sum(out.max(1)[1] == data.y[data.train_mask]) / out.shape[0]

    if epoch % args.print_freq == 0:
        print(f'epoch: {epoch}, train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

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


def real_train(args, __dataset, __data, checkpoint_dir=None):
    dataset = copy.deepcopy(__dataset)
    data = copy.deepcopy(__data)

    percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(args.val_rate * len(data.y)))
    TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)
    print('True Label rate: ', TrueLBrate)

    args.C = len(data.y.unique())
    args.Gamma = None

    # test_acc, best_val_acc, Gamma_0 = RunExp(args, dataset, data, Net, percls_trn, val_lb)

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

    # 这段是什么作用？？
    # if checkpoint_dir:
    #     model_state, optimizer_state = torch.load(
    #         os.path.join(checkpoint_dir, "checkpoint"))
    #     model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)

    for epoch in range(args.epochs):
        epoch_loss, epoch_acc = model_train(model, optimizer, data, args.dprate, epoch)

        [tmp_train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = model_test(model, data)

        # 这边确实用到验证集了，保留验证集合最好的数据
        if val_loss < best_val_loss:#val_acc > best_val_acc:#
            best_val_loss = val_loss
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            print("test_middle*:", test_acc,", val_middle:", best_val_acc,", val_loss:",best_val_loss)
            # train_acc = tmp_train_acc
            # best_model_wts = copy.deepcopy(model.state_dict())

        # 提前停止
        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
            # if train_acc > 0.99:
            #     break

        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((model.state_dict(), optimizer.state_dict()), path)
    # model.load_state_dict(best_model_wts)
    return optimizer, test_acc
    # tune.report(accuracy=test_acc)
    # print("Finished Training")
    #
    # print("train_acc:", train_acc)
    # print("best_val_acc:", best_val_acc, ",best_val_loss", best_val_loss)
    # print("test_acc:", test_acc)
    # print(train_acc, best_val_acc, test_acc)
    # model.load_state_dict(best_model_wts)


def trainable_large(config):
    args.lr = config['lr']
    args.alpha = config['alpha']
    args.weight_decay = config['weight_decay']
    args.dprate = config['dropout']
    args.dropout = config['dropout']
    args.early_stopping = int(config['early_stopping'])*100

    test_list = []
    for i in range(args.RPMAX):
        _, test_acc = real_train(args, ray.get(dataset_id), ray.get(data_id))
        test_list.append(test_acc)

    # 参数reporter主要是把训练标准传递给tune进行优化，报告用于调度，搜索或提前停止的指标。
    tune.report(mean_accuracy=np.mean(test_list))


    # print("train_acc:", train_acc)
    # print("best_val_acc:", best_val_acc, ",best_val_loss", best_val_loss)
    print("test_acc:", np.mean(test_list), ", lr:", config["lr"], ", alpha:", config["alpha"])
    print("============Finished Training============")
    # print("best_optimizer——lr+++:", config["lr"], "alpha+++:", config["alpha"])
    # print(train_acc, best_val_acc, test_acc)
    # model.load_state_dict(best_model_wts)

def trainable(config, dataset, data):
    args.lr = config['lr']
    args.alpha = config['alpha']
    #args.weight_decay = config['weight_decay']
    args.dprate = config['dp']
    args.dropout = config['dropout']
    args.early_stopping = int(config['early_stopping'])*100

    test_list = []
    for i in range(args.RPMAX):
        _, test_acc = real_train(args, dataset, data)
        test_list.append(test_acc)

    # 参数reporter主要是把训练标准传递给tune进行优化，报告用于调度，搜索或提前停止的指标。
    tune.report(mean_accuracy=np.mean(test_list))


    # print("train_acc:", train_acc)
    # print("best_val_acc:", best_val_acc, ",best_val_loss", best_val_loss)
    print("test_acc:", np.mean(test_list), ", lr:", config["lr"], ", alpha:", config["alpha"])
    print("============Finished Training============")
    # print("best_optimizer——lr+++:", config["lr"], "alpha+++:", config["alpha"])
    # print(train_acc, best_val_acc, test_acc)
    # model.load_state_dict(best_model_wts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='film')  # 更改数据集
    parser.add_argument('--RPMAX', type=int, default=50)  # 重复执行次数,10
    parser.add_argument('--epochs', type=int, default=1000)  # 1000
    parser.add_argument('--early_stopping', type=int, default=200)  # 200

    parser.add_argument('--Order', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.002)            #0.002
    parser.add_argument('--alpha', type=float, default=0.1)          #0.1
    parser.add_argument('--weight_decay', type=float, default=0.00)#0.0005
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
    parser.add_argument('--net', type=str,
                        choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN', 'HeGCN', 'HiSCN','HiGCN','HeGCN_exp_new'],
                        default='HiGCN')

    args = parser.parse_args()

    args.lr = 0.01
    args.alpha = 0.9
    args.weight_decay = 0.0
    args.dprate = 0.7

    dataset, data = DataLoader(args.dataset, args)
    ray.init()
    Net = get_net(args.net)

    gnn_name = args.net


    # config = {  # 设定需要搜索的超参空间
    #     "threads": 2,
    #     "early_stopping": tune.uniform(2, 7),
    #     # "weight_decay": tune.choice([0.0, 0.0005, 0.001]),
    #     # "alpha": tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    #     # "dp": tune.choice([0.0, 0.3, 0.5, 0.7, 0.9]),
    #     # "dropout": tune.choice([0.0, 0.3, 0.5, 0.7, 0.9]),
    #     "lr": tune.loguniform(1e-2, 8e-1),
    #     #"weight_decay": tune.loguniform(1e-8, 5e-3),
    #     "alpha": tune.uniform(0.0, 1.0),
    #     "dp": tune.uniform(0.0, 0.9),
    #     "dropout": tune.uniform(0.0, 0.9),
    # }
    config = {  # 设定需要搜索的超参空间
        "threads": 2,
        "early_stopping": tune.choice([5]),
        "lr": tune.choice([0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7]),#[0.001,0.005,0.01,0.05,0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
        "weight_decay": tune.choice([0.0, 0.0001, 0.0005, 0.001, 0.005, 0.1]),
        "alpha": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0]),
        "dropout": tune.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]),
    }
    # 选择超参优化器及指定其参数
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        # metric="mean_accuracy",
        # mode="max",
        max_t=1000,  # 每次试验的最大时间单位，在 max_t 时间单位（由 time_attr 确定）过去后，试验将停止。
        grace_period=20)  # 只停止至少这么久的试验？？还是没明白

    # 在CLIReporter中指定的指标，将会取代reporter函数的指定内容，但并不是完全覆盖，因为CLIReporter中指定的指标只有在reporter中被赋予值后才会在状态表中显示。
    reporter = CLIReporter(metric_columns=["loss", "mean_accuracy", "training_iteration"])

    # 大数据集时把下面两行取消注释，再用partial(trainable_large),
    dataset_id = ray.put(dataset)
    data_id = ray.put(data)


    tune.TuneConfig(reuse_actors=True)
    # 执行run程序，并且同时指定各种参数（可选）其中tune.run返回值为result，可用analysis接受以便后续分析
    analysis = tune.run(
        partial(trainable_large),
        # partial(trainable, dataset=dataset, data=data),
        local_dir='./results',  # 文件保存路径
        name="exp_" + args.dataset + "_hid32_622_",  # experiment的名称，与输出结果后的路径有关
        metric='mean_accuracy',  # 最后比较的指标
        mode='max',  # 指标越大越好
        scheduler=sched,
        #search_alg=BayesOptSearch(random_search_steps=6),#贝叶斯优化来进行调参，默认是通过网格化生成参数，并从网格中随机选取参数，random_search_steps初始化随机搜索参数，这对于避免贝叶斯过程的初始局部过拟合是必要的
        num_samples=300,  # 总共运行Trails的数目。从超参数空间采样的次数
        resources_per_trial={"cpu": 8, "gpu": 1},  # 每个Trail可支配的计算资源
        config=config,
        progress_reporter=reporter,
        # stop={  # 每次试验将在完成 training_iteration 次迭代或达到 mean_accuracy 的平均准确度时停止。
        #     "mean_accuracy": 0.99,
        #     "training_iteration": 3  # 这个到底什么作用？？
        # }  # 设定提前终止trail的条件，
    )

    # best_trial = analysis.best_trial  # Get best trial
    # best_config = analysis.best_config  # Get best trial's hyperparameters
    # best_logdir = analysis.best_logdir  # Get best trial's logdir
    # best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
    # best_result = analysis.best_result  # Get best trial's last results

    # （训练数据会自动保存至local_dir下，如："~/ray_results/exp"后面附录中有关于结果数据的详细讲解）
    # 输出最佳结果

    print("Best config is:", analysis.get_best_config(metric="mean_accuracy", mode="max"))
    print("Best result is:", analysis.best_result['mean_accuracy'])
    #best_trial = analysis.get_best_trial("loss", "min", "last")

