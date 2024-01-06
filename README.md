# HiGCN

Welcome to the source code repository for our paper: **Higher-order Graph Convolutional Network with Flower-Petals Laplacians on Simplicial Complexes**


<p align="center">
  <img src=".\FP_model.png" width="700">
</p>

The simplicial complexes (SCs) are a powerful tool to model higher-order interactions with elegant and potent mathematical tools. 
We creatively construct a higher-order **flower-petals (FP) model** and introduce **FP Laplacians** to study SCs.
Additionally, we propose a higher-order graph convolutional network (**HiGCN**) based on the FP Laplacians, which can capture intrinsic topology features at different scales.

# Citation
Please cite our paper if you use this code in your own work:
```latex
@inproceedings{HiGCN2024,
  title={Higher-Order Graph Convolutional Network with Flower-Petals Laplacians on Simplicial Complexes},
  author={Huang, Yiming and Zeng, Yujie and Wu, Qiang and L{\"u}, Linyuan},
  year={2024},
  booktitle={Proceedings of the AAAI conference on artificial intelligence (AAAI)},
  url = {https://arxiv.org/abs/2309.12971},
}
```



# Prerequisites:
Ensure you have the following libraries installed:
```
pytorch
pytorch-geometric
networkx
numpy
```
# Exp1: Node Classification 
go to folder `./node_classify/src`

## Run experiment with Cora:


```sh
cd node_classify/src
python train_model.py --RPMAX 100 \
        --net HiGCN \
        --dataset cora \
        --lr 0.1 \
        --alpha 0.5 \
        --weight_decay 0.001 \
        --dprate 0.3
```

Before the training commences, the script will download and preprocess the respective graph datasets. 
Subsequently, it performs the appropriate graph-lifting procedure (this process might a while).

## Repreduce results in Table 2:


For details of optimization of all models, please refer to **Appendix I** of our paper. Here are the settings for HiGCN:

We choose random walk path lengths with K = 10 and use a 2-layer (MLP) with 32 hidden units for the NN component. 



Here is a list of hyperparameters for your reference:
We find that learning rate (lr) and weight decay (wd) has huge influence on the results.

- For cora, we choose lr = 0.1, alpha = 0.5, wd = 0.001, dprate = 0.3. 
- For citeseer, we choose lr = 0.05, alpha = 0.7, wd = 0.1, dropout = 0.2, dprate = 0.2.
- For pubmed, we choose lr = 0.05, alpha = 0.8, wd = 0.005, dprate = 0.2.
- For computers, we choose lr = 0.1, alpha = 0.0, wd = 0.0, dprate = 0.7.
- For Photo, we choose lr = 0.1, alpha = 0.3, wd = 0.0, dropout = 0.3, dprate = 0.6.
- For chameleon, we choose lr = 0.1, alpha = 0.4, wd = 0.0, dropout = 0.8, dprate = 0.7, early_stopping = 500.
- For Actor, we choose lr = 0.01, alpha = 0.5, wd = 0.001, dropout = 0.8, dprate = 0.8, early_stopping = 500.
- For squirrel, we choose lr = 0.3, alpha = 1.0, wd = 0.0,  dropout = 0.6, dprate = 0.6, early_stopping = 500.
- For Texas, we choose lr = 0.1, alpha = 0.8, dropout = 0.6, dprate = 0.6, early_stopping = 500.
- For Wisconsin, we choose lr = 0.2, alpha = 0.7, dropout = 0.6, dprate = 0.6, early_stopping = 500.

We provide more hyperparameter details on the ``Reproduce_HiGCN.sh``. 
If some hyperparameter are not given above, it is same as the default value in the train_model.py.


## Repreduce results in Table 3:
### Create null model for dataset Texas:
For academic confidentiality, we can't release the code which is used to generate the null model, 
so we publish the edge list of each null model in the folder `nullModel_Texas` under `data` folder.

If you want to run null model dataset, just change Dataset name as Texas_null and choose rho in 
`['-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0.1', '0.2', '0.3', '0.4', '0.5']` (mentiond: `--rho` need to be string)
`rho=0.5` means the network has 50% plus triangles of the origin network.

```sh
python train_model.py --RPMAX 100 --net HiGCN --dataset Texas_null --lr 0.1 --alpha 0.5 --weight_decay 0.001 --dprate 0.3 --rho='0.1'
```

# Exp2: graph classification
go to folder `./graph_classify`

We prepared individual scripts for each experiment. The results are written in the
`exp/results/` directory and are also displayed in the terminal once the training is
complete. 
```shell
cd graph_classify
sh scripts/HiGCN-PROETINS.sh
```



# Exp3: simplicial data imputation
go to folder `./CoSCs`


The results will be displayed in the terminal. 
To run an experiment on coauthor complexes with HiGCN, execute the following command:
```shell
cd CoSCs
sh CoSCs-HiGCN.sh
```







 
Thank you for your interest in our work. If you have any questions or encounter any issues while using our code, please don't hesitate to raise an issue or reach out to us directly.


