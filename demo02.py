"""
【GNN框架系列】DGL第二讲：使用Deep Graph Library实现GNN进行链接预测
https://blog.csdn.net/feilong_csdn/article/details/117150872?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171383646016800186529890%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=171383646016800186529890&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-117150872-null-null.nonecase&utm_term=DGL&spm=1018.2226.3001.4450
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp

import dgl.data
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# 随机抽取数据集中10%的边作为测试集中的正样例，剩下的90%数据集中的边作为训练集，然后随机为训练集和测试集
# 负采样生成相同数量的负样例，使得训练集和测试集中的正负样本比例为1:1
# 将数据集中边的集合划分到训练集和测试集中，训练集90%，测试集10%
u, v = g.edges()
eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.1)
train_size = g.number_of_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# 采样所有负样例并划分为训练集和测试集中。
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)

neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

# 需要注意：在模型训练时，需要将图中在测试集中10%的边移除掉，防止数据泄露，使用dgl.remove_edges
train_g = dgl.remove_edges(g, eids[:test_size])

from dgl.nn import SAGEConv
# 定义一个两层的GraphSage模型
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# 模型通过定义函数来预测两个节点表示之间的得分，从而来判断两个节点之间存在边的可能性
# 在GNN节点分类任务重，模型是训练得到单个节点的表征，但在链接计算任务中是预测节点对的表征
# 在给出节点对的预测得分函数之前，先需要理解一下：
# DGL使用方式是先将节点对视为一个图，同时一条边可用来描述一对节点。
# 在链接预测中，会得到一个正图，它包含所有的正例子作为边，以及一个负图，它包含所有的负例子。
# 正图和负图将包含与原始图相同的节点集。这使得在多个图中传递节点特征更容易进行计算。
# 可以直接将在整个图上计算的节点表示形式提供给正图和负图，用于计算节点对的两两得分。
# 将节点对视为图的好处是可以使用DGLGraph.apply_edges方法，
# 基于节点的特征表示和原始图中边的特征表示可以方便地计算新产生边的特征表示。
# DGL提供了一组优化的内置函数，可以直接使用原始节点/边特征表示计算新的边特征表示

train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

# 接下来定义两个节点之间的得分函数预测，可以直接使用DGL提供的，也可以自定义，
#下面DotPredictor是官方提供的预测函数，MLPPredictor是自定义的预测函数。
import dgl.function as fn

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # 通过源节点特征“h”和目标节点特征“h”之间的点积计算两点之间存在边的Score
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v为每条边返回一个元素向量，因此需要squeeze操作
            return g.edata['score'][:, 0]

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

model = GraphSAGE(train_g.ndata['feat'].shape[1], 16)
# 可以使用自定义的MLPPredictor代替DotPredictor
#pred = MLPPredictor(16)
pred = DotPredictor()

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

all_logits = []
for e in range(100):
    # 前向传播
    h = model(train_g, train_g.ndata['feat'])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # 后向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))

# 检测结果准确性
from sklearn.metrics import roc_auc_score
with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print('AUC', compute_auc(pos_score, neg_score))