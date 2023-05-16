# 文件说明

1. data
* 每个数据集一个文件夹('DBLP', 'History', 'Geology'这几个单纯形数据集在原始数据上进行了加工处理)
  * raw原始数据文件夹
  * SCNN 存放SNN和SCNN运行需要的矩阵（通过运行SCNN_dataprocess.py获得）
  * name-node-labels.txt: 新节点编号 旧节点编号 作者
* data_filter.py 对原始数据集进行过滤(原始数据集太大了,可以通过tri_threshold, start_year 两个参数控制生成的单纯复形的规模)
* data_process.py 计算网络的特征，生成HL
* SCNN_dataprocess.py 为SNN和SCNN生成需要的矩阵

3. models
   * co_models.py 存放我们的HiGCN模型
   * SCNNs.py 存放了SNN和SCNN模型

4. utils
   * coDataLoader.py 存放了HiGCN的dataLoader
   * SCNN_loaders.py存放了SNN和SCNN的dataLoader

