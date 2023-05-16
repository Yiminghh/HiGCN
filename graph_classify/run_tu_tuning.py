import itertools
import os
import copy
import yaml
import argparse
from definitions import ROOT_DIR
from exp.parser import get_parser
from exp.run_tu_exp import exp_main

__max_devices__ = 1

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='tuning.')
    parser.add_argument('--conf', type=str, default='./tuning_configurations/tu_template.yml',help='path to yaml configuration')
    parser.add_argument('--code', type=str, help='tuning name')
    parser.add_argument('--idx', type=int, default=0, help='selection index') #这个后面好像赋予了device
    t_args = parser.parse_args()
    
    # parse grid from yaml
    with open(t_args.conf, 'r') as handle:
        conf = yaml.safe_load(handle)


    dataset = conf['dataset']
    hyper_list = list()
    hyper_values = list()
    for key in conf:
        if key == 'dataset':
            continue
        hyper_list.append(key)
        hyper_values.append(conf[key])
    grid = itertools.product(*hyper_values)#*符号在这里用于解压可迭代对象并将其作为函数参数
    exp_queue = list()#创建一个实验队列，用于存储当前设备（GPU）正在运行的实验任务。
    for h, hypers in enumerate(grid):
        if h % __max_devices__ == (t_args.idx % __max_devices__):
            exp_queue.append((h, hypers))
    
    # form args
    result_folder = os.path.join(ROOT_DIR, 'exp', 'results', '{}_tuning_{}_new_new'.format(dataset, conf['model'][0]))
    base_args = [
        '--device', str(t_args.idx),
        '--task_type', 'classification',
        '--eval_metric', 'accuracy',
        '--dataset', dataset,
        '--result_folder', result_folder]
    
    for exp in exp_queue:
        args = copy.copy(base_args)
        addendum = ['--exp_name', str(exp[0])]
        hypers = exp[1]
        for name, value in zip(hyper_list, hypers):
            addendum.append('--{}'.format(name))
            addendum.append('{}'.format(value))
            if name == 'lr_scheduler_decay_rate':
                if value == 0.5:
                    addendum.append('--lr_scheduler_decay_steps')
                    addendum.append('50')
                elif value == 0.9:
                    addendum.append('--lr_scheduler_decay_steps')
                    addendum.append('20')

        args += addendum
        exp_main(args)
        # mean_perf, msg, final_args = exp_main(args)
        # filename = os.path.join(result_folder, f'{final_args.dataset}-{final_args.exp_name}-{mean_perf}','result.txt')
        # if not os.path.exists(filename):
        #     os.makedirs(filename)
        # print('Writing results at: {}'.format(filename))
        # with open(filename, 'w') as handle:
        #     handle.write(msg)


