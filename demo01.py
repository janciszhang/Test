"""
【GNN框架系列】DGL第一讲：使用Deep Graph Library实现GNN进行节点分类
https://blog.csdn.net/feilong_csdn/article/details/117150681
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.data
# 加在dgl库提供的Cora数据对象
dataset = dgl.data.CoraGraphDataset()
print('Number of categories:', dataset.num_classes)

# DGL库中Dataset数据集可能是包含多个图的，但Cora数据集是由单个图组成
g = dataset[0]

# DGL库中一个Graph是使用字典的形式存储了Node Features和Edge Features，
# g.ndata使用字典结构存储了节点特征信息
# g.edata使用字典结构存储了边特征信息
# 对于Cora数据集的graph来说，Node Features包含以下五个方面
# 1、train_mask: 指示节点是否在训练集中的布尔张量
# 2、val_mask: 指示节点是否在验证集中的布尔张量
# 3、test_mask: 指示节点是否在测试机中的布尔张量
# 4、label: 每个节点的真实类别
# 5、feat: 节点自身的属性
print('Node features: ', g.ndata)
print('Edge features: ', g.edata)

# 使用dgl库中的dgl.nn.GraphConv模块构建了一个两层GCN网络，每层都通过汇聚邻居节点信息来更新节点表征
# 每层GCN网络存在维度的变化，第一层维度映射(in_feats, h_feats)，
# 第二层维度映射(h_feats, num_classes)，两层网络因此第二层直接映射到最终分类类别维度上。
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        # 这里g代表的Cora数据Graph信息，一般就是经过归一化的邻接矩阵
        # in_feat表示的是node representation，即节点初始化特征信息
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# 使用给定的维度创建GCN模型，其中hidden维度设定为16，输入维度和输出维度由数据集确定。
model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(100):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
train(g, model)

# 上述代码是在CPU上进行训练，如果在GPU上训练需要将数据和模型转化到cuda上。
# g = g.to('cuda')
# model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes).to('cuda')
# train(g, model)