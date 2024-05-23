import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

# 加载Cora数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN()

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

"""
No Partition GNN
"""

import time

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

start_time = time.time()

for epoch in range(200):
    loss = train(model, data, optimizer)
    if epoch % 10 == 0:
        train_acc, val_acc, test_acc = test(model, data)
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

no_partition_time = time.time() - start_time
no_partition_accuracy = test(model, data)[2]

print(f"No Partition - Time: {no_partition_time:.4f}s, Test Accuracy: {no_partition_accuracy:.4f}")



"""
Partition GNN
"""
# 进行图分割，这里使用Kernighan-Lin分割法进行均匀分割
G = nx.Graph()
G.add_edges_from(data.edge_index.t().tolist())
partition = kernighan_lin_bisection(G)

# 分割成子图
subgraph_1_nodes = list(partition[0])
subgraph_2_nodes = list(partition[1])

subgraph_1 = data.subgraph(torch.tensor(subgraph_1_nodes))
subgraph_2 = data.subgraph(torch.tensor(subgraph_2_nodes))

# 分别对每个子图训练模型
def train_subgraph(subgraph):
    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(200):
        loss = train(model, subgraph, optimizer)
    return model

start_time = time.time()

model_1 = train_subgraph(subgraph_1)
model_2 = train_subgraph(subgraph_2)

partition_time = time.time() - start_time

# 测试两个子图的模型在整个图上的性能
def combined_test(model_1, model_2, data):
    model_1.eval()
    model_2.eval()
    logits_1 = model_1(data)
    logits_2 = model_2(data)
    combined_logits = (logits_1 + logits_2) / 2  # 简单平均
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = combined_logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

partition_accuracy = combined_test(model_1, model_2, data)[2]

print(f"Partition - Time: {partition_time:.4f}s, Test Accuracy: {partition_accuracy:.4f}")


print(f"No Partition - Time: {no_partition_time:.4f}s, Test Accuracy: {no_partition_accuracy:.4f}")
print(f"Partition - Time: {partition_time:.4f}s, Test Accuracy: {partition_accuracy:.4f}")

