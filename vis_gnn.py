"""
ref: https://blog.csdn.net/qq_35831906/article/details/134024843
NetworkX和Matplotlib
NetworkX是一个Python库，用于创建、操作和可视化图结构。
Matplotlib则是用于绘制图的流行绘图库。
"""
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edge('A', 'B')
G.add_edge('B', 'C')
G.add_edge('C', 'D')
G.add_edge('D', 'A')

pos = nx.spring_layout(G)  # 使用Spring布局算法定义节点位置
nx.draw(G, pos, with_labels=True, node_size=500, font_size=10, node_color='lightblue', font_color='black')
plt.title("GNN Small Sample")
plt.show()

"""
可视化图结构
可以使用NetworkX和Matplotlib等库可视化图结构，
但通常需要添加节点和边的属性信息以更好地了解图的结构。
"""
import networkx as nx
import matplotlib.pyplot as plt

# 创建一个有向图
G = nx.DiGraph()

# 添加节点并设置节点属性
G.add_node('A', color='red', size=100)
G.add_node('B', color='blue', size=200)
G.add_node('C', color='green', size=150)
G.add_node('D', color='yellow', size=250)

# 添加边并设置边属性
G.add_edge('A', 'B', weight=5)
G.add_edge('A', 'C', weight=3)
G.add_edge('B', 'D', weight=7)
G.add_edge('C', 'D', weight=2)

# 设置节点的位置（可选）
pos = nx.spring_layout(G)

# 绘制节点
node_colors = [G.nodes[node]['color'] for node in G]
node_sizes = [G.nodes[node]['size'] for node in G]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

# 绘制边
edge_weights = [G.edges[edge]['weight'] for edge in G.edges]
nx.draw_networkx_edges(G, pos, width=edge_weights)

# 添加标签
node_labels = {node: node for node in G}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black')

# 显示图
plt.axis('off')
plt.title("GNN Sample with Attributes")
plt.show()


"""
可视化节点属性
在图神经网络中，每个节点通常具有属性。
可以使用颜色、大小等视觉特征来表示这些属性。
"""

import networkx as nx
import matplotlib.pyplot as plt

# 创建一个简单图
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5), (4, 6)])

# 计算每个节点的度数，并将其作为节点属性
degree_values = dict(G.degree())

# 为每个节点分配一个颜色，颜色的深浅与节点度数成正比
node_colors = [degree_values[node] for node in G.nodes]

# 创建一个虚拟的图像对象以便创建颜色条
img = plt.imshow([[0, 1]], cmap=plt.cm.Reds)
img.set_visible(False)  # 隐藏虚拟图像

# 定义节点位置
pos = nx.spring_layout(G)

# 绘制图
nx.draw(G, pos, node_color=node_colors, cmap=plt.cm.Reds, with_labels=True, node_size=500)

# 添加颜色条以显示节点属性值与颜色之间的映射关系
plt.colorbar(img, ax=plt.gca(), orientation='vertical', label='Node Degree')

plt.title("GNN Sample in Different Color")
plt.show()

"""
可视化中间表示
使用PyTorch Geometric库和Matplotlib可视化GNN模型的中间表示
"""
import torch
import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np

# 创建一个简单的图数据
edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                      [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
x = torch.randn(5, 16)  # 随机节点特征

data = Data(x=x, edge_index=edges)

# 创建一个简单的GNN模型
class SimpleGNN(torch.nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(16, 32)
        self.conv2 = pyg_nn.GCNConv(32, 64)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

gnn_model = SimpleGNN()

# 获取中间表示
intermediate_output = gnn_model(data)

# 提取节点位置（这里只是一个示例，你可以根据需要计算节点的位置）
pos = np.random.rand(5, 2)  # 随机位置

# 提取节点颜色
node_colors = intermediate_output.detach().numpy().sum(axis=1)

# 创建图和热图的子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 绘制图（节点和边）
for i, j in edges.t().numpy():
    ax1.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 'k-', alpha=0.5)

# 绘制节点
sc = ax1.scatter(pos[:, 0], pos[:, 1], c=node_colors, cmap='viridis', s=300)
ax1.set_title("Graph Structure")
fig.colorbar(sc, ax=ax1, label='Node Feature Sum')

# 绘制热图
im = ax2.imshow(intermediate_output.detach().numpy(), cmap='viridis', aspect='auto')
ax2.set_title("Intermediate Representation Heatmap")
ax2.set_xlabel('Feature Dimension')
ax2.set_ylabel('Node Index')
fig.colorbar(im, ax=ax2, label='Feature Value')

plt.show()



"""
不使用 torch_geometric 库，
使用NetworkX 创建一个简单的图，
然后使用 PyTorch 和 Matplotlib 可视化 GNN 模型的中间表示
"""
import torch
import networkx as nx
import matplotlib.pyplot as plt

# 创建一个简单的图
G = nx.Graph()
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
G.add_edges_from(edges)

# 随机生成节点特征
num_nodes = len(G.nodes)
node_features = torch.randn(num_nodes, 16)

# 创建一个简单的GNN模型
class SimpleGNN(torch.nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        self.conv1 = torch.nn.Linear(16, 32)
        self.conv2 = torch.nn.Linear(32, 64)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x

gnn_model = SimpleGNN()

# 获取中间表示
intermediate_output = gnn_model(node_features, edges)

# 可视化中间表示
# plt.figure(figsize=(8, 6))
fig, ax = plt.subplots(figsize=(8, 6))
pos = nx.spring_layout(G)
node_colors = intermediate_output.detach().numpy().sum(axis=1)  # 使用detach()来获得不需要梯度的张量
# nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap='viridis', node_size=1000)
nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap='viridis', node_size=1000, ax=ax)
plt.title("Intermediate Representation via NetworkX")

# 创建ScalarMappable对象
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=node_colors.min(), vmax=node_colors.max()))
# sm._A = []  # 这里必须设置_A为一个空列表，以便创建颜色条
sm.set_array([])  # 设置为一个空数组

# 添加颜色条
# cbar = plt.colorbar(sm)
cbar = plt.colorbar(sm, ax=ax)

plt.show()