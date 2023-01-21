# HiGCN

This is the source code for our  paper: **Higher-order Graph Convolutional Network with Flower-Petals Laplacians on Simplicial Complexes**


<p align="center">
  <img src=".\img\FP_model.png" width="700">
</p>

[//]: # (Hidden state feature extraction is performed by a neural networks using individual node features propagated via GPR. Note that both the GPR weights <img src="https://render.githubusercontent.com/render/math?math=\gamma_k"> and parameter set <img src="https://render.githubusercontent.com/render/math?math=\{\theta\}"> of the neural network are learned simultaneously in an end-to-end fashion &#40;as indicated in red&#41;.)
The simplicial complex is a powerful tool to model higher-order interactions with elegant and potent mathematical tools. 
We propose a hierarchical simplicial conventional network (HiGCN) via higher-order random walk based Laplacian, which can capture intrinsic topology features at different scales based on different dimension simplices.



# Requirement:
```
pytorch
pytorch-geometric
networkx
numpy
```

# Run experiment with Cora:

go to folder `src`
```
python train_model.py --RPMAX 100 \
        --net HiGCN \
        --dataset cora \
        --lr 0.1 \
        --alpha 0.5 \
        --weight_decay 0.001 \
        --dprate 0.3
```

# Repreduce results in Table 2:


For details of optimization of all models, please refer to **Appendix G** of our paper. Here are the settings for HiGCN:

| Method  |       Cora       |     Citeseer     |      PubMed      |     Computers    |       Photo      |     Chameleon    |       Actor      |     Squirrel     |       Texas       |     Wisconsin    |
|:-------:|:----------------:|:----------------:|:----------------:|:----------------:|:----------------:|:----------------:|:----------------:|:----------------:|:-----------------:|:----------------:|
|   MLP   | 76.96 $\pm$ 0.95 | 76.58 $\pm$ 0.88 | 85.94 $\pm$ 0.22 | 82.85 $\pm$ 0.38 | 84.72 $\pm$ 0.34 | 46.85 $\pm$ 1.51 | 40.19 $\pm$ 0.56 | 31.03 $\pm$ 1.18 | 91.45 $\pm$ 1.14  | 93.56 $\pm$ 0.87 |
|   GCN   | 87.14 $\pm$ 1.01 | 79.86 $\pm$ 0.67 | 86.74 $\pm$ 0.27 | 83.32 $\pm$ 0.33 | 88.26 $\pm$ 0.73 | 60.96 $\pm$ 0.78 | 30.59 $\pm$ 0.23 | 45.66 $\pm$ 0.39 | 75.16 $\pm$ 0.96  | 61.93 $\pm$ 0.83 |
|   GAT   | 88.03 $\pm$ 0.79 | 80.52 $\pm$ 0.71 | 87.04 $\pm$ 0.24 | 83.33 $\pm$ 0.38 | 90.94 $\pm$ 0.68 | 63.90 $\pm$ 0.46 | 35.98 $\pm$ 0.23 | 42.72 $\pm$ 0.33 | 78.87 $\pm$ 0.86  | 65.64 $\pm$ 1.74 |
| ChebNet | 86.67 $\pm$ 0.82 | 79.11 $\pm$ 0.75 | 87.95 $\pm$ 0.28 | 87.54 $\pm$ 0.43 | 93.77 $\pm$ 0.32 | 59.96 $\pm$ 0.51 | 38.02 $\pm$ 0.23 | 40.67 $\pm$ 0.31 | 86.08 $\pm$ 0.96  | 90.57 $\pm$ 0.91 |
| BernNet | 88.52 $\pm$ 0.95 | 80.09 $\pm$ 0.79 | 88.48 $\pm$ 0.41 | 87.64 $\pm$ 0.44 | 93.63 $\pm$ 0.35 | 68.29 $\pm$ 1.58 | 41.79 $\pm$ 1.91 | 51.35 $\pm$ 0.73 | 93.12 $\pm$ 0.65  | 91.82 $\pm$ 0.38 |
|  GGCN   | 87.68 $\pm$ 1.26 | 77.08 $\pm$ 1.32 | 89.63 $\pm$ 0.46 |        OOM       | 89.92 $\pm$ 0.97 | 62.72 $\pm$ 2.05 | 38.09 $\pm$ 0.88 | 49.86 $\pm$ 1.55 | 85.81 $\pm$ 1.72  | 87.65 $\pm$ 1.50 |
|  APPNP  | 88.14 $\pm$ 0.73 | 80.47 $\pm$ 0.74 | 88.12 $\pm$ 0.31 | 85.32 $\pm$ 0.37 | 88.51 $\pm$ 0.31 | 51.89 $\pm$ 1.82 | 39.66 $\pm$ 0.55 | 34.71 $\pm$ 0.57 | 90.98 $\pm$ 1.64  | 64.59 $\pm$ 0.97 |
| GPRGNN  | 88.57 $\pm$ 0.69 | 80.12 $\pm$ 0.83 | 88.46 $\pm$ 0.33 | 86.85 $\pm$ 0.25 | 93.85 $\pm$ 0.28 | 67.28 $\pm$ 1.09 | 39.92 $\pm$ 0.67 | 50.15 $\pm$ 1.92 | 92.95 $\pm$ 1.31  | 88.54 $\pm$ 1.37 |
| 1-HiGCN | 88.96 $\pm$ 0.24 | 80.96 $\pm$ 0.27 | 89.83 $\pm$ 0.73 | 90.50 $\pm$ 0.52 | 95.22 $\pm$ 0.10 | 63.55 $\pm$ 0.84 | 41.57 $\pm$ 0.27 | 49.13 $\pm$ 0.33 | 90.36 $\pm$ 0.78  | 94.39 $\pm$ 0.94 |
| 2-HiGCN | 89.33 $\pm$ 0.23 | 81.12 $\pm$ 0.28 | 89.89 $\pm$ 0.16 | 90.76 $\pm$ 0.11 | 95.33 $\pm$ 0.15 | 68.36 $\pm$ 0.38 | 41.81 $\pm$ 0.52 | 51.86 $\pm$ 0.42 | 92.03 $\pm$ 0.73  | 94.45 $\pm$ 0.95 |


We choose random walk path lengths with K = 10 and use a 2-layer (MLP) with 32 hidden units for the NN component. 



Here is a list of hyperparameters for your reference:
We find that learning rate (lr) and weight decay (wd) has huge influence on the results.

- For cora, we choose lr = 0.1, alpha = 0.5, wd = 0.001, dprate = 0.3. 
- For citeseer, we choose lr = 0.05, alpha = 0.7, wd = 0.1, dropout = 0.2, dprate = 0.2.
- For pubmed, we choose lr = 0.05, alpha = 0.8, wd = 0.005, dprate = 0.2.
- For computers, we choose lr = 0.1, alpha = 0.0, wd = 0.0, dprate = 0.7.
- For Photo, we choose lr = 0.1, alpha = 0.3, wd = 0.0, dropout = 0.3, dprate = 0.6.
- For chameleon, we choose lr = 0.08, alpha = 0.4, wd = 0.0, dropout = 0.65, dprate = 0.85, early_stopping = 400.
- For Actor, we choose lr = 0.03, alpha = 0.4, wd = 0, dropout = 0.9, dprate = 0.7, early_stopping = 500.
- For squirrel, we choose lr = 0.3, alpha = 1.0, wd = 0.0,  dropout = 0.6, dprate = 0.6, early_stopping = 500.
- For Texas, we choose lr = 0.1, alpha = 0.5, wd = 0.0001, dropout = 0.7, dprate = 0.7.
- For Wisconsin, we choose lr = 0.2, alpha = 0.7, wd = 0.0005, dropout = 0.6, dprate = 0.2, early_stopping = 400.

We provide more hyperparameter details on the Reproduce_HiGCN.sh. 
If some hyperparameter are not given above, it is same as the default value in the train_model.py.


# Repreduce results in Table 3:
## Create null model for dataset Texas:
For academic confidentiality, we can't release the code which is used to generate the null model, 
so we publish the edge list of each null model in the folder `nullModel_Texas` under `data` folder.

If you want to run null model dataset, just change Dataset name as Texas_null and choose rho in 
`['-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0.1', '0.2', '0.3', '0.4', '0.5']` (mentiond: `--rho` need to be string)
`rho=0.5` means the network has 50% plus triangles of the origin network.

```
python train_model.py --RPMAX 100 --net HiGCN --dataset Texas_null --lr 0.1 --alpha 0.5 --weight_decay 0.001 --dprate 0.3 --rho='0.1'
```


# Simplified generate Laplacian matrix function
Due to the complexity and memory limitation, we offer the simplified version of generating higher-order Laplacian matrix. ^_^

We use the simplified function on the squirrel dataset, and other dataset use the origin function.

# Citation
Please cite our paper if you use this code in your own work:
```latex

```

 



