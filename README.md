# HiGCN

Welcome to the source code repository for our paper: **Higher-order Graph Convolutional Network with Flower-Petals Laplacians on Simplicial Complexes**

[Yiming Huang](https://yimingh.top/), [Yujie Zeng](http://yujie.world/), [Qiang Wu](https://scholar.google.co.uk/citations?hl=en&user=edUqF7sAAAAJ&view_op=list_works&sortby=pubdate), [Linyuan Lü](https://linyuanlab.com/)

[Vedio]()

<p align="center">
  <img src=".\FP_model.png" width="700">
</p>

The simplicial complexes (SCs) are a powerful tool to model higher-order interactions with elegant and potent mathematical tools. 
We creatively construct a higher-order **flower-petals (FP) model** and introduce **FP Laplacians** to study SCs.
Additionally, we propose a higher-order graph convolutional network (**HiGCN**) based on the FP Laplacians, which can capture intrinsic topology features at different scales.





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
python run_node_exp.py --RPMAX 100 \
        --net HiGCN \
        --dataset cora \
        --lr 0.1 \
        --alpha 0.5 \
        --weight_decay 0.001 \
        --dprate 0.3
```

Before the training commences, the script will download and preprocess the respective graph datasets. 
Subsequently, it performs the appropriate graph-lifting procedure (this process might a while).





## Create null model for dataset Texas:
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




# Citation
Please cite our work if you find our code/paper is useful to your work. :
```latex
@inproceedings{HiGCN2024,
  title={Higher-Order Graph Convolutional Network with Flower-Petals Laplacians on Simplicial Complexes},
  author={Huang, Yiming and Zeng, Yujie and Wu, Qiang and L{\"u}, Linyuan},
  year={2024},
  booktitle={Proceedings of the AAAI conference on artificial intelligence (AAAI)},
  url = {https://arxiv.org/abs/2309.12971},
}
```


 
Thank you for your interest in our work. If you have any questions or encounter any issues while using our code, please don't hesitate to raise an issue or reach out to us directly.


