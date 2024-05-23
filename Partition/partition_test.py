from torch_geometric.datasets import Planetoid
import torch
import networkx as nx

# 加载Cora数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]


import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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

import community as community_louvain
import metis
from scipy.sparse.linalg import eigsh
import numpy as np

def to_networkx(data):
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    G.add_edges_from(edge_index.T)
    for i in range(data.num_nodes):
        G.add_node(i, x=data.x[i].numpy())
    return G

def from_networkx(G, original_data):
    new_data = original_data.clone()
    node_map = {n: i for i, n in enumerate(G.nodes())}
    new_data.edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in G.edges()]).T.contiguous()
    new_data.x = torch.tensor([G.nodes[n]['x'] for n in G.nodes()])
    new_data.train_mask = original_data.train_mask[list(G.nodes())]
    new_data.val_mask = original_data.val_mask[list(G.nodes())]
    new_data.test_mask = original_data.test_mask[list(G.nodes())]
    new_data.y = original_data.y[list(G.nodes())]
    return new_data

def louvain_partition(data):
    G = to_networkx(data)
    partition = community_louvain.best_partition(G)
    subgraphs = {}
    for node, part in partition.items():
        if part not in subgraphs:
            subgraphs[part] = []
        subgraphs[part].append(node)
    return [from_networkx(G.subgraph(nodes), data) for nodes in subgraphs.values()]

def metis_partition(data, nparts):
    G = to_networkx(data)
    _, parts = metis.part_graph(G, nparts=nparts)
    subgraphs = {i: [] for i in range(nparts)}
    for node, part in enumerate(parts):
        subgraphs[part].append(node)
    return [from_networkx(G.subgraph(nodes), data) for nodes in subgraphs.values()]

def spectral_partition(data):
    G = to_networkx(data)
    L = nx.normalized_laplacian_matrix(G).astype(float)
    _, eigenvectors = eigsh(L, k=2, which='SM')
    second_smallest_eigenvector = eigenvectors[:, 1]
    partition = second_smallest_eigenvector > 0
    subgraph_1 = G.subgraph([n for i, n in enumerate(G.nodes()) if partition[i]])
    subgraph_2 = G.subgraph([n for i, n in enumerate(G.nodes()) if not partition[i]])
    return [from_networkx(subgraph_1, data), from_networkx(subgraph_2, data)]


def train_subgraph(subgraph):
    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(200):
        loss = train(model, subgraph, optimizer)
    return model

def combined_test(models, data):
    logits_list = [model(data) for model in models]
    combined_logits = sum(logits_list) / len(logits_list)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = combined_logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

import time

# 原始图（不分割）
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

start_time = time.time()

for epoch in range(200):
    loss = train(model, data, optimizer)

no_partition_time = time.time() - start_time
no_partition_accuracy = test(model, data)[2]

print(f"No Partition - Time: {no_partition_time:.4f}s, Test Accuracy: {no_partition_accuracy:.4f}")

# Louvain分割
start_time = time.time()
subgraphs = louvain_partition(data)
models = [train_subgraph(subgraph) for subgraph in subgraphs]
louvain_time = time.time() - start_time
louvain_accuracy = combined_test(models, data)[2]

print(f"Louvain - Time: {louvain_time:.4f}s, Test Accuracy: {louvain_accuracy:.4f}")

# Metis分割
start_time = time.time()
subgraphs = metis_partition(data, nparts=4)
models = [train_subgraph(subgraph) for subgraph in subgraphs]
metis_time = time.time() - start_time
metis_accuracy = combined_test(models, data)[2]

print(f"Metis - Time: {metis_time:.4f}s, Test Accuracy: {metis_accuracy:.4f}")

# Spectral分割
start_time = time.time()
subgraphs = spectral_partition(data)
models = [train_subgraph(subgraph) for subgraph in subgraphs]
spectral_time = time.time() - start_time
spectral_accuracy = combined_test(models, data)[2]

print(f"Spectral - Time: {spectral_time:.4f}s, Test Accuracy: {spectral_accuracy:.4f}")

print(f"No Partition - Time: {no_partition_time:.4f}s, Test Accuracy: {no_partition_accuracy:.4f}")
print(f"Louvain - Time: {louvain_time:.4f}s, Test Accuracy: {louvain_accuracy:.4f}")
print(f"Metis - Time: {metis_time:.4f}s, Test Accuracy: {metis_accuracy:.4f}")
print(f"Spectral - Time: {spectral_time:.4f}s, Test Accuracy: {spectral_accuracy:.4f}")
