from run_coauthor_exp import RunExp, parse_args
import itertools
from utils.coDataLoader import coDataLoader
from utils.SCNN_loaders import SNNLoader, SCNNLoader
from models.co_models import co_HiGCN
from models.SCNNs import SNN, SCNN
import numpy as np
import os
from datetime import datetime

if __name__ == '__main__':
    lr_list = [0.001, 0.005,  0.01, 0.05]
    wd_list = [0.0]#,0.001,0.01,0.0001

    sample_pro = [0.2,0.4]
    repeat_times = 10
    args = parse_args()
    graph_list = ['History','DBLP','Geology']#'History', 'DBLP',, 'DBLP','Geology',
    model_list = ['SNN']#,', ,,'SNN','HiGCN'

    for sp, graph,  net in itertools.product(sample_pro, graph_list,  model_list):
        args.dataset = graph
        args.train_rate = sp
        args.net = net
        log_path = os.path.join('.', 'results',f'{graph}-{args.net}', f'sample{sp}')
        os.makedirs(log_path, exist_ok=True)
        if net == 'HiGCN':
            alpha_list = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9,1.0]
        else:
            alpha_list = [0.0]

        best_corr = -999
        for lr, alpha, wd in itertools.product(lr_list, alpha_list, wd_list):
            args.lr, args.alpha, args.weight_decay = lr, alpha, wd
            corr_history = []
            for exp_id in range(repeat_times):
                if args.net == 'HiGCN':
                    data = coDataLoader(args, printData=False)
                    model = co_HiGCN(data, args)
                elif args.net == 'SNN':
                    data = SNNLoader(args)
                    model = SNN(data, args)
                elif args.net == 'SCNN':
                    data = SCNNLoader(args)
                    model = SCNN(data, args)
                corr, _ = RunExp(args, data, model, print_log=False)
                corr_history.append(corr)
                print(f'exp_id:{exp_id}, corr:{corr}, mean_corr:{np.mean(corr_history)}')


            if np.mean(corr_history)>= best_corr:
                best_corr = np.mean(corr_history)
                best_corr_std = np.std(corr_history)
                best_lr = lr
                best_alpha = alpha
                best_wd = wd

            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            log = f'{current_time}(best_acc{best_corr}), lr:{lr}, alpha:{alpha}, wd:{wd},\
                  corr={np.mean(corr_history)}({np.std(corr_history)}) \n'
            print(log)
            with open(os.path.join(log_path,'results_log.txt'), 'a+', buffering=1) as f:
                f.write(log)
                f.flush()

        best_log = f'best_acc{best_corr}(std:{best_corr_std}), lr:{best_lr}, alpha:{best_alpha}, wd:{best_wd} \n'
        print("best_log:", best_log)
        with open(os.path.join(log_path, 'best_log.txt'), 'a+', buffering=1) as f:
            f.write(best_log)
            f.flush()

    print("Finish!")




