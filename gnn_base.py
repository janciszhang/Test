"""
ref: https://zhuanlan.zhihu.com/p/380734367

torch CPU or GPU install
pip3 install torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

CPU版本包安装
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

GPU版本包安装
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-geometric

"""
import torch
# print(torch.__version__) # 检查版本
import torch_scatter
import torch_sparse
import torch_cluster
import torch_spline_conv
from torch_geometric.datasets import KarateClub


"""PyG中图的表示及使用（Data类） 
通过观察PyG中内置的一个图先测试一下PyG是否安装成功，同时查看一下图数据Data对象的性质"""
dataset = KarateClub()
data = dataset[0]  # Get the first graph object.
# print(data)
# print('==============================================================')
#
# # 获取图的一些信息
# print(f'Number of nodes: {data.num_nodes}') # 节点数量
# print(f'Number of edges: {data.num_edges}') # 边数量
# print(f'Number of node features: {data.num_node_features}') # 节点属性的维度
# print(f'Number of node features: {data.num_features}') # 同样是节点属性的维度
# print(f'Number of edge features: {data.num_edge_features}') # 边属性的维度
# print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}') # 平均节点度
# print(f'if edge indices are ordered and do not contain duplicate entries.: {data.is_coalesced()}') # 是否边是有序的同时不含有重复的边
# print(f'Number of training nodes: {data.train_mask.sum()}') # 用作训练集的节点
# print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}') # 用作训练集的节点的数量
# print(f'Contains isolated nodes: {data.contains_isolated_nodes()}') # 此图是否包含孤立的节点
# print(f'Contains self-loops: {data.contains_self_loops()}')  # 此图是否包含自环的边
# print(f'Is undirected: {data.is_undirected()}')  # 此图是否是无向图


"""PyG中图数据集的表示及使用（Dataset类）
PyG内置了大量常用的基准数据集，接下来以PyG内置的Planetoid数据集为例，来学习PyG中图数据集Dataset的表示及使用。"""
# 生成数据集对象并分析数据集
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='./data', name='Cora')
# print(dataset)# Cora()
# print(len(dataset))# 1
# print(dataset.num_classes)# 7
# print(dataset.num_node_features)# 1433

data = dataset[0]
# print(data) # Data(edge_index=[2, 10556], test_mask=[2708],train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
# print(data.is_undirected())# True
# print(data.train_mask.sum().item())# 140
# print(data.val_mask.sum().item())# 500
# print(data.test_mask.sum().item())# 1000

# 现在我们看到该数据集包含的唯一的图，有2708个节点，节点特征为1433维，有10556条边，有140个用作训练集的节点，有500个用作验证集的节点，有1000个用作测试集的节点。

# 数据集的使用
# 假设我们定义好了一个图神经网络模型，其名为Net。在下方的代码中，我们展示了节点分类图数据集在训练过程中的使用。
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# class Net(nn.Module):
#     def __init__(self, num_features, num_classes):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(num_features, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(32 * 28 * 28, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(-1, 32 * 28 * 28)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
#
#
# # device = torch.device('cuda:0') # tensor a is in CPU
# # Usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Net(num_features=1, num_classes=10).to(device)
# data = dataset[0].to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#
# model.train()
# for epoch in range(200):
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()



"""
图数据的代码实现
前一部分只是单纯使用现成包中的数据，在这一部分中将会通过代码手动构建数据的类，并结合实际问题进行分析。
"""

# 4.1 Data类的构建
# Data类的构造函数：
class Data(object):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, **kwargs):
        r"""
        Args:
            x (Tensor, optional): 节点属性矩阵，大小为`[num_nodes, num_node_features]`
            edge_index (LongTensor, optional): 边索引矩阵，大小为`[2, num_edges]`，第0行为尾节点，第1行为头节点，头指向尾
            edge_attr (Tensor, optional): 边属性矩阵，大小为`[num_edges, num_edge_features]`
            y (Tensor, optional): 节点或图的标签，任意大小（，其实也可以是边的标签）
        """
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y

        for key, item in kwargs.items():
            if key == 'num_nodes':
                self.__num_nodes__ = item
            else:
                self[key] = item

    graph_dict = {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'y': y,
        'num_nodes': num_nodes,
        'other_attr': other_attr
    }
    graph_data = Data.from_dict(graph_dict)



    @classmethod
    def from_dict(cls, dictionary):
        r"""Creates a data object from a python dictionary."""
        data = cls()
        for key, item in dictionary.items():
            data[key] = item

        return data

    # 将Data对象转换为dict对象：
    def to_dict(self):
        return {key: item for key, item in self}

    # 获取Data对象属性
    x = graph_data['x']

    # 设置Data对象属性
    graph_data['x'] = x

    # 获取Data对象包含的属性的关键字
    graph_data.keys()

    # 对边排序并移除重复的边
    graph_data.coalesce()


# Data类的构造函数：
'''
【OAP，机构-作者-论文】
O-Orginazation,机构；
A-Author,作者；
P-Paper,论文
'''


class OAP_Data(object):
    def __init__(self, x_O=None, x_A=None, x_P=None, edge_index_A_O=None, edge_index_A_P=None, edge_attr=None, y=None,
                 **kwargs):
        r"""
        Args:
            x_O (Tensor, optional): 节点属性矩阵，大小为`[num_nodes_O, num_node_O_features]`
            x_A (Tensor, optional): 节点属性矩阵，大小为`[num_nodes_A, num_node_A_features]`
            x_P (Tensor, optional): 节点属性矩阵，大小为`[num_nodes_P, num_node_P_features]`
            edge_index_A_O (LongTensor, optional): 边索引矩阵，大小为`[2, num_edges_A_O]`，第0行为尾节点，第1行为头节点，头指向尾
            edge_index_A_P (LongTensor, optional): 边索引矩阵，大小为`[2, num_edges_A_P]`，第0行为尾节点，第1行为头节点，头指向尾
            edge_attr_A_O (Tensor, optional): 边属性矩阵，大小为`[num_edges_A_O, 1]`  # 边没有属性，故列为1
            edge_attr_A_P (Tensor, optional): 边属性矩阵，大小为`[num_edges_A_P, 1]`  # 边没有属性，故列为1
            y (Tensor, optional): 节点或图的标签，任意大小（也可以是边的标签）
        """
        self.x_O = x_O  # 机构类节点
        self.x_A = x_A  # 作者类节点
        self.x_P = x_P  # 论文类节点
        self.edge_index_A_O = edge_index_A_O  # 作者-机构边的序号
        self.edge_index_A_P = edge_index_A_P  # 作者-论文边的序号
        # 边没有属性
        self.edge_attr_A_O = edge_attr_A_O  # 作者-机构边的属性
        self.edge_attr_A_P = edge_attr_A_P  # 作者-论文边的属性
        self.y = y  # 标签

    # 实例方法
    def num_nodes_O(self):
        return self.x_O.shape[0]  # 机构节点数量

    def num_nodes_A(self):
        return self.x_A.shape[0]  # 作者节点数量

    def num_nodes_P(self):
        return self.x_P.shape[0]  # 论文节点数量

    def num_edges_A_O(self):
        return self.edge_index_A_O.shape[1]  # 作者-机构边数量

    def num_edges_A_P(self):
        return self.edge_index_A_P.shape[1]  # 作者-论文边数量

    @classmethod
    def from_dict(cls, dictionary):
        r"""Creates a data object from a python dictionary."""
        data = cls()
        for key, item in dictionary.items():
            data[key] = item

        return data




# 输入原始数据
OAP_graph_dict = {
    'x_O': x_O,
    'x_A': x_A,
    'x_P': x_P,
    'edge_index_A_O': edge_index_A_O,
    'edge_index_A_P': edge_index_A_P,
    'edge_attr_A_O': edge_attr_A_O,
    'edge_attr_A_P': edge_attr_A_P,
    'y': y,
    'other_attr': other_attr
}

# 转dict对象为Data对象
OAP_graph_data = OAP_Data.from_dict(OAP_graph_dict)

# 获取OAP图上不同节点、不同边的数量
print(f'Number of orginazation nodes：{OAP_graph_data.num_nodes_O}') # 节点数量
print(f'Number of author nodes：{OAP_graph_data.num_nodes_A}') # 机构数量
print(f'Number of paper nodes：{OAP_graph_data.num_nodes_P}') # 论文数量
print(f'Number of author-orginazation edges：{OAP_graph_data.num_edges_A_O}') # 作者-机构边数量
print(f'Number of author-paper edges： {OAP_graph_data.num_edges_A_P}') # 作者-论文边数量

