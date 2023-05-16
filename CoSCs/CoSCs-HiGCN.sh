#! /bin/sh

python run_coauthor_exp.py --net HiGCN --dataset Geology --train_rate 0.1 --lr 0.05 --alpha 0.1 --weight_decay 5e-4

python run_coauthor_exp.py --net HiGCN --dataset History --train_rate 0.1 --lr 0.05 --alpha 0.1 --weight_decay 0.0

python run_coauthor_exp.py --net HiGCN --dataset DBLP --train_rate 0.1 --lr 0.01 --alpha 0.9 --weight_decay 5e-4