# HiGCN

This is the source code for our  paper: **Higher-order Graph Convolutional Network with Flower-Petals Laplacians on Simplicial Complexes**


<p align="center">
  <img src=".\img\bipartiteG3.png" width="600">
</p>

[//]: # (Hidden state feature extraction is performed by a neural networks using individual node features propagated via GPR. Note that both the GPR weights <img src="https://render.githubusercontent.com/render/math?math=\gamma_k"> and parameter set <img src="https://render.githubusercontent.com/render/math?math=\{\theta\}"> of the neural network are learned simultaneously in an end-to-end fashion &#40;as indicated in red&#41;.)
The simplicial complex is a powerful tool to model higher-order interactions with elegant and potent mathematical tools. 
We propose a hierarchical simplicial conventional network (Hi-SCN) via higher-order random walk based Laplacian, which can capture intrinsic topology features at different scales based on different dimension simplices.


[//]: # (The learnt GPR weights of the GPR-GNN on real world datasets. Cora is homophilic while Texas is heterophilic &#40;Here, H stands for the level of homophily defined in the main text, Equation &#40;1&#41;&#41;. An interesting trend may be observed: For the heterophilic case the weights alternate from positive to negative with dampening amplitudes. The shaded region corresponds to a 95% confidence interval.)


[//]: # (<p align="center">)

[//]: # (  <img src="https://github.com/jianhao2016/GPRGNN/blob/master/figs/Different_gamma_upated_H.png" width="600">)

[//]: # (</p>)

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
        --net HiSCN \
        --dataset cora \
        --lr 0.1 \
        --alpha 0.5 \
        --weight_decay 0.001 \
        --dprate 0.3
```

# Repreduce results in Table 2:

To reproduce the results in Table 2 of [our paper](https://openreview.net/forum?id=n6jl7fLxrP) you need to first perform hyperparameter tuning. 
For details of optimization of all models, please refer to section D.2 in Appendix of our paper. Here are the settings for HiSCN:

We choose random walk path lengths with K = 10 and use a 2-layer (MLP) with 32 hidden units for the NN component. 

[//]: # (For the GPR weights, we use different initializations including PPR with <img src="https://render.githubusercontent.com/render/math?math=\alpha\in\{0.1, 0.2, 0.5, 0.9\}">, <img src="https://render.githubusercontent.com/render/math?math=\gamma_k=\delta_{0k}"> or <img src="https://render.githubusercontent.com/render/math?math=\delta_{Kk}"> and the default random initialization in pytorch. Similarly, for APPNP we search the optimal <img src="https://render.githubusercontent.com/render/math?math=\alpha\in\{0.1, 0.2, 0.5, 0.9\}">. For other hyperparameter tuning, we optimize the learning rate over {0.002, 0.01, 0.05} and weight decay {0.0, 0.0005} for all models. )

<!-- <img src="https://render.githubusercontent.com/render/math?math=\alpha\in\{0.1, 0.2, 0.5, 0.9\}">
<img src="https://render.githubusercontent.com/render/math?math=\gamma_k=\delta_{0k}">
<img src="https://render.githubusercontent.com/render/math?math=\delta_{Kk}"> -->


Here is a list of hyperparameters for your reference:
We find that learning rate (lr) and weight decay (wd) has huge influence on the results.

- For cora, we choose lr = 0.1, alpha = 0.5, wd = 0.001, dprate = 0.3. 
- For citeseer, we choose lr = 0.05, alpha = 0.7, wd = 0.1, dropout = 0.2, dprate = 0.2.
- For pubmed, we choose lr = 0.05, alpha = 0.8, wd = 0.005, dprate = 0.2.
- For computers, we choose lr = 0.1, alpha = 0.0, wd = 0.0, dprate = 0.7.
- For Photo, we choose lr = 0.1, alpha = 0.3, wd = 0.0, dropout = 0.3, dprate = 0.6.
- For chameleon, we choose lr = 0.08, alpha = 0.4, wd = 0, dropout = 0.65, dprate = 0.85, early_stopping = 400.
- For Actor, we choose lr = 0.03, alpha = 0.4, wd = 0, dropout = 0.9, dprate = 0.7, early_stopping = 500.
- For squirrel, we choose lr = 0.5, alpha = 0.7, wd = 0, early_stopping = 500.
- For Texas, we choose lr = 0.1, alpha = 0.5, wd = 0.0001, dropout = 0.7, dprate = 0.7.
- For Wisconsin, we choose lr = 0.2, alpha = 0.7, wd = 0.0005, dropout = 0.6, dprate = 0.2, early_stopping = 400.

We provide more hyperparameter details on the Reproduce_HiSCN.sh. 
If some hyperparameter are not given above, it is same as the default value in the train_model.py.


# Repreduce results in Table 3:
## Create null model for Texas:
For academic confidentiality, we can't release the code which is used to generate the null model, 
so we publish the edge list of each null model in the folder `nullModel_Texas` under `data` folder.

If you want to run null model dataset, just change Dataset name as Texas_null and choose rho in 
`['-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0.1', '0.2', '0.3', '0.4', '0.5']` (mentiond: `--rho` need to be string)
`rho=0.5` means the network has 50% plus triangles of the origin network.

```
python train_model.py --RPMAX 100 --net HiSCN --dataset Texas_null --lr 0.1 --alpha 0.5 --weight_decay 0.001 --dprate 0.3 --rho='0.1'
```


# Simplified generate Laplacian matrix function
Due to the complexity and memory limitation, we offer the simplified version of generating higher-order Laplacian matrix. ^_^

We use the simplified function on the squirrel dataset, and other dataset use the origin function.

# Citation
Please cite our paper if you use this code in your own work:
```latex

```

 



