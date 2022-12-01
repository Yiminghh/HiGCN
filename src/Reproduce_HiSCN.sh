#! /bin/sh

# Below is for homophily datasets

python train_model.py --RPMAX 100 --net HiSCN --dataset cora --lr 0.1 --alpha 0.5 --weight_decay 0.001 --dprate 0.3

python train_model.py --RPMAX 100 --net HiSCN --dataset citeseer --lr 0.05 --alpha 0.7 --weight_decay 0.1 --dprate 0.2 --dropout 0.2

python train_model.py --RPMAX 100 --net HiSCN --dataset pubmed --lr 0.05 --alpha 0.8 --weight_decay 0.005 --dprate 0.2
        
python train_model.py --RPMAX 100 --net HiSCN --dataset computers --lr 0.1 --alpha 0.0 --weight_decay 0.0 --dprate 0.7
        
python train_model.py --RPMAX 100 --net HiSCN --dataset photo --lr 0.1 --alpha 0.3 --weight_decay 0.0 --dprate 0.6 --dropout 0.3


# Below is for heterophily datasets

python train_model.py --RPMAX 100 --net HiSCN --dataset chameleon --lr 0.08 --alpha 0.4 --weight_decay 0.0 --dprate 0.85 --dropout 0.65 --early_stopping 400

python train_model.py --RPMAX 100 --net HiSCN --dataset film --lr 0.03 --alpha 0.4 --weight_decay 0.0 --dprate 0.7 --dropout 0.9 --early_stopping 500

python train_model.py --RPMAX 100 --net HiSCN --dataset squirrel --lr 0.5 --alpha 0.7 --weight_decay 0.0 --early_stopping 500
# python train_model.py --RPMAX 100 --net HiSCN --dataset squirrel --lr 0.5 --alpha 0.7 --weight_decay 0.0  --dprate 0.5 --dropout 0.7 --early_stopping 500
# python train_model.py --RPMAX 100 --net HiSCN --dataset squirrel --lr 0.5 --alpha 0.7 --weight_decay 0.0  --dprate 0.7 --dropout 0.7 --early_stopping 300


python train_model.py --RPMAX 100 --net HiSCN --dataset texas --lr 0.1 --alpha 0.5 --weight_decay 0.0001 --dprate 0.7 --dropout 0.7

python train_model.py --RPMAX 100 --net HiSCN --dataset wisconsin --lr 0.2 --alpha 0.7 --weight_decay 0.0005 --dprate 0.6 --dropout 0.2 --early_stopping 400

