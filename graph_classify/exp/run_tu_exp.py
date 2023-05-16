import sys
import os
import copy
import time
import numpy as np
from exp.parser import get_parser
from exp.run_exp import main
import datetime

# python3 -m exp.run_tu_exp --dataset IMDBBINARY --model cin --drop_rate 0.0 --lr 0.0001 --max_dim 2 --emb_dim 32 --dump_curves --epochs 30 --num_layers 1 --lr_scheduler StepLR --lr_scheduler_decay_steps 5

__num_folds__ = 10
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'#仅在调试时使用

def print_summary(summary):
    msg = ''
    for k, v in summary.items():
        msg += f'Fold {k:1d}:  {v:.3f}\n'
    print(msg)
    
    
def exp_main(passed_args):
    
    parser = get_parser()
    args = parser.parse_args(copy.copy(passed_args))

    # run each experiment separately and gather results
    results = list()
    total_time = []
    for i in range(__num_folds__):
        print("@@@@@@ num_folds:", i)
        current_args = copy.copy(passed_args) + ['--fold', str(i)]
        parsed_args = parser.parse_args(current_args)
        start_time = time.time()
        curves = main(parsed_args)
        end_time = time.time()
        total_time.append(end_time - start_time)
        results.append(curves)


        
    # aggregate results
    val_curves = np.asarray([curves['val'] for curves in results])
    avg_val_curve = val_curves.mean(axis=0)
    best_index = np.argmax(avg_val_curve)
    mean_perf = avg_val_curve[best_index]
    std_perf = val_curves.std(axis=0)[best_index]

    print(" ===== Mean performance per fold ======")
    perf_per_fold = val_curves.mean(1)
    perf_per_fold = {i: perf_per_fold[i] for i in range(len(perf_per_fold))}
    print_summary(perf_per_fold)

    print(" ===== Max performance per fold ======")
    perf_per_fold = np.max(val_curves, axis=1)
    perf_per_fold = {i: perf_per_fold[i] for i in range(len(perf_per_fold))}
    print_summary(perf_per_fold)

    print(" ===== Median performance per fold ======")
    perf_per_fold = np.median(val_curves, axis=1)
    perf_per_fold = {i: perf_per_fold[i] for i in range(len(perf_per_fold))}
    print_summary(perf_per_fold)

    print(" ===== Performance on best epoch ======")
    perf_per_fold = val_curves[:, best_index]
    perf_per_fold = {i: perf_per_fold[i] for i in range(len(perf_per_fold))}
    print_summary(perf_per_fold)

    total_time = np.array(total_time)
    print(" ===== Final result ======")
    msg = (
        f'Dataset:        {args.dataset}\n'
        f'Accuracy:       {mean_perf} ± {std_perf}\n'
        f'Best epoch:     {best_index}\n'
        f'total time:     {np.mean(total_time)} ± {np.std(total_time)}\n'
        '-------------------------------\n')
    print(msg)
    
    # additionally write msg and configuration on file
    msg += str(args)
    old_filename = os.path.join(args.result_folder, f'{args.dataset}-{args.exp_name}')
    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(args.result_folder, f'{args.dataset}-{args.exp_name}-{mean_perf:.4f}-({cur_time})')
    os.rename(old_filename, filename)#使用os.rename()函数修改文件夹名称
    if not os.path.exists(filename):
        os.makedirs(filename)
    filename = os.path.join(filename, 'result.txt')
    print('Writing results at: {}'.format(filename))
    with open(filename, 'w') as f:
        f.write(msg)



def gin_args(args):
    args += ['--model', 'gin']
    args += ['--device', '0']
    args+= ['--start_seed', '0']
    args+= ['--stop_seed', '9']
    args+= ['--exp_name', 'mpsn-redditb']
    args+= ['--dataset', 'REDDITBINARY']
    args+= ['--train_eval_period', '50']
    args+= ['--epochs', '200']
    args+= ['--batch_size', '32']
    args+= ['--drop_rate', '0.0']
    args+= ['--drop_position', 'final_readout']
    args+= ['--emb_dim', '64']#隐藏层维度
    args+= ['--max_dim', '2']
    args+= ['--final_readout', 'sum']
    args+= ['--init_method', 'mean']
    args += ['--jump_mode', 'cat']
    args+= ['--lr', '0.001']
    args+= ['--graph_norm', 'id']
    args+= ['--nonlinearity', 'relu']
    args+= ['--num_layers', '4']
    args+= ['--readout', 'sum']
    args+= ['--task_type', 'classification']
    args+= ['--eval_metric', 'accuracy']
    args+= ['--lr_scheduler', 'StepLR']
    args += ['--lr_scheduler_decay_rate', '0.5']
    args += ['--lr_scheduler_decay_steps', '50']
    args+= ['--use_coboundaries', 'False']
    args+= ['--preproc_jobs', '4']
    return args

def cwn_args(args):
    args += ['--model', 'sparse_cin']
    args += ['--device', '0']
    args+= ['--start_seed', '0']
    args+= ['--stop_seed', '9']
    args+= ['--exp_name', 'cwn-nci109']
    args+= ['--dataset', 'NCI109']
    args+= ['--train_eval_period', '50']
    args+= ['--epochs', '150']
    args+= ['--batch_size', '32']
    args+= ['--drop_rate', '0.0']
    args+= ['--drop_position', 'lin2']
    args+= ['--emb_dim', '64']#隐藏层维度
    args+= ['--max_dim', '2']
    args+= ['--final_readout', 'sum']
    args+= ['--init_method', 'mean']
    args += ['--jump_mode', 'cat']
    args+= ['--lr', '0.001']
    args+= ['--graph_norm', 'bn']
    args+= ['--nonlinearity', 'relu']
    args+= ['--num_layers', '4']
    args+= ['--readout', 'sum']
    args += ['--max_ring_size', '6']
    args+= ['--task_type', 'classification']
    args+= ['--eval_metric', 'accuracy']
    args+= ['--lr_scheduler', 'StepLR']
    args += ['--lr_scheduler_decay_rate', '0.5']
    args += ['--lr_scheduler_decay_steps', '20']
    args+= ['--use_coboundaries', 'True']
    args+= ['--preproc_jobs', '4']
    return args

def my_args(args):
    args += ['--model', 'HiGCN_gin']#sparse_cin, HiGCN_gin
    args += ['--petalType', "simplex"]#simplex,ring
    args += ['--max_petal_dim', "2"]
    args+= ['--exp_name', 'HiGCN'] #1:mpsn-redditb,2:HiGCN-NCI1,3:gin-NCI1 4:HiGCN-MUTAG 5:HiGCN-PTC, 6:HiGCN-NCI109
    args+= ['--dataset', 'NCI109'] #1:REDDITBINARY, 2:NCI1, 3:NCI1 4:MUTAG, 5:PTC, 6:NCI109， 7:IMDBBINARY 8：IMDBMULTI,9:PROTEINS, REDDITMULTI5K
    args+= ['--epochs', '300']#200
    args+= ['--batch_size', '64']#32
    args += ['--emb_dim', '32']  # 隐藏层维度
    args+= ['--drop_rate', '0.0']
    args += ['--lr', '0.001']  # 0.001
    args += ['--lr_scheduler_decay_steps', '20']
    args += ['--lr_scheduler_decay_rate', '0.9']

    args += ['--num_layers', '1']  # 3,2
    args += ['--readout', 'sum']#mean
    args+= ['--final_readout', 'sum']
    args += ['--train_eval_period', '50']

    args += ['--max_dim', '2']
    args += ['--graph_norm', 'bn'] #bn, id
    args += ['--preproc_jobs', '2']
    args+= ['--init_method', 'mean']
    args += ['--jump_mode', 'cat']

    args+= ['--nonlinearity', 'relu']
    args += ['--drop_position', 'lin2']
    args+= ['--task_type', 'classification']
    args+= ['--eval_metric', 'accuracy']
    args+= ['--lr_scheduler', 'StepLR']
    args+= ['--use_coboundaries', 'False']

    args += ['--device', '0']
    args+= ['--start_seed', '0']
    args+= ['--stop_seed', '9']
    return args


def mpsn_args(args):
    args += ['--model', 'sparse_cin']#sparse_cin, HiGCN_gin
    args += ['--petalType', "simplex"]#simplex,ring
    args += ['--max_petal_dim', "2"]
    args+= ['--exp_name', 'MPSN-MUTAG'] #1:mpsn-redditb,2:HiGCN-NCI1,3:gin-NCI1 4:HiGCN-MUTAG 5:HiGCN-PTC, 6:HiGCN-NCI109
    args+= ['--dataset', 'MUTAG'] #1:REDDITBINARY, 2:NCI1, 3:NCI1 4:MUTAG, 5:PTC, 6:NCI109， 7:IMDBBINARY 8：IMDBMULTI,9:PROTEINS, REDDITMULTI5K
    args+= ['--epochs', '200']#200
    args+= ['--batch_size', '32']#32
    args += ['--emb_dim', '16']  # 隐藏层维度
    args+= ['--drop_rate', '0.0']
    args += ['--lr', '0.001']  # 0.001
    args += ['--lr_scheduler_decay_steps', '50']
    args += ['--lr_scheduler_decay_rate', '0.5']

    args += ['--num_layers', '4']  # 3,2
    args += ['--readout', 'sum']#mean
    args+= ['--final_readout', 'sum']
    args += ['--train_eval_period', '50']

    args += ['--max_dim', '2']
    args += ['--graph_norm', 'bn'] #bn, id
    args += ['--preproc_jobs', '2']
    args+= ['--init_method', 'mean']
    args += ['--jump_mode', 'cat']

    args+= ['--nonlinearity', 'relu']
    args += ['--drop_position', 'lin2']
    args+= ['--task_type', 'classification']
    args+= ['--eval_metric', 'accuracy']
    args+= ['--lr_scheduler', 'StepLR']
    args+= ['--use_coboundaries', 'False']

    args += ['--device', '0']
    args+= ['--start_seed', '0']
    args+= ['--stop_seed', '9']
    return args

def COLLAB_args(args):
    args += ['--model', 'HiGCN_gin']#sparse_cin, HiGCN_gin
    args += ['--petalType', "simplex"]#simplex,ring
    args += ['--max_petal_dim', "2"]
    args+= ['--exp_name', 'HiGCN_gin'] #1:mpsn-redditb,2:HiGCN-NCI1,3:gin-NCI1 4:HiGCN-MUTAG 5:HiGCN-PTC, 6:HiGCN-NCI109
    args+= ['--dataset', 'COLLAB'] #1:REDDITBINARY, 2:NCI1, 3:NCI1 4:MUTAG, 5:PTC, 6:NCI109， 7:IMDBBINARY 8：IMDBMULTI,9:PROTEINS, REDDITMULTI5K
    args+= ['--train_eval_period', '30']
    args+= ['--epochs', '200']#200
    args+= ['--batch_size', '64']#32
    args += ['--emb_dim', '32']  # 隐藏层维度
    args+= ['--drop_rate', '0.0']
    args += ['--lr', '0.001']  # 0.001
    args += ['--lr_scheduler_decay_steps', '50']
    args += ['--lr_scheduler_decay_rate', '0.5']

    args+= ['--max_dim', '2']
    args += ['--readout', 'mean']#mean
    args+= ['--final_readout', 'sum']
    args += ['--num_layers', '1']  # 3,2

    args+= ['--init_method', 'mean']
    args += ['--jump_mode', 'cat']
    args+= ['--graph_norm', 'id']
    args+= ['--nonlinearity', 'relu']
    args += ['--drop_position', 'lin2']
    args+= ['--task_type', 'classification']
    args+= ['--eval_metric', 'accuracy']
    args+= ['--lr_scheduler', 'StepLR']
    args+= ['--use_coboundaries', 'False']
    args+= ['--preproc_jobs', '4']
    args += ['--device', '0']
    args+= ['--start_seed', '0']
    args+= ['--stop_seed', '9']
    return args


if __name__ == "__main__":
    
    # standard args
    passed_args = sys.argv[1:]
    assert 'fold' not in passed_args
    #passed_args = my_args(passed_args)
    #passed_args = mpsn_args(passed_args)
    #passed_args  = cwn_args(passed_args)
    exp_main(passed_args)
