"""
export METIS_DLL=/opt/homebrew/opt/metis/lib/libmetis.dylib
"""
import time
import networkx as nx
import metis
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# 创建一个简单的图
G = nx.karate_club_graph()
edges = list(G.edges())
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# 定义一个简单的GNN模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 参数设置
in_channels = 1
hidden_channels = 16
out_channels = 2
lr = 0.01
epochs = 100
K_values = [k for k in range(1,10)]  # 不同的分割数量K

# 训练和评估模型
accuracies = []
training_times = []
for K in K_values:
    start_time = time.time()

    # 使用Metis进行图分割
    _, parts = metis.part_graph(graph=G, nparts=K)

    # 构建图数据
    x = torch.tensor(parts, dtype=torch.float).view(-1, 1)
    y = torch.tensor([0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in G.nodes()], dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index)

    # 初始化模型和优化器
    model = GCN(in_channels, hidden_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

    # 评估模型
    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=1)
        accuracy = (pred == data.y).sum().item() / len(pred)
        accuracies.append(accuracy)

    end_time = time.time()
    training_times.append(end_time - start_time)

# 可视化结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(K_values, accuracies, marker='o')
plt.xlabel('K values')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs. K values')

plt.subplot(1, 2, 2)
plt.plot(K_values, training_times, marker='o')
plt.xlabel('K values')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time vs. K values')

plt.tight_layout()
plt.show()
