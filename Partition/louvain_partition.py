import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from networkx.algorithms.community import greedy_modularity_communities

# 创建一个简单的图，并为每个节点添加 'feature' 和 'label' 属性
G = nx.karate_club_graph()
for node in G.nodes:
    G.nodes[node]['feature'] = np.random.rand(10)  # 生成长度为 10 的随机特征向量
    G.nodes[node]['label'] = np.random.randint(0, 2)  # 生成随机标签 0 或 1

# 定义Louvain算法，返回分割后的子图列表
def louvain_partition(graph, k):
    partitions = greedy_modularity_communities(graph)
    return list(partitions)[:k]

# 定义函数来训练GNN模型并返回准确性
def train_gnn(partitions):
    accuracies = []
    for partition in partitions:
        nodes = list(partition)
        X = np.array([G.nodes[node]['feature'] for node in nodes])  # 获取节点特征向量
        y = np.array([G.nodes[node]['label'] for node in nodes])  # 获取节点标签

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        accuracies.append(accuracy)
    return accuracies

# 探究不同分割数量K下的运行速度和模型准确性
k_values = [k for k in range(1,10)]  # 不同的分割数量
execution_times = []
accuracies_list = []

for k in k_values:
    start_time = time.time()
    partitions = louvain_partition(G, k)
    accuracies = train_gnn(partitions)
    end_time = time.time()
    execution_times.append(end_time - start_time)
    accuracies_list.append(accuracies)

# 可视化结果
plt.figure(figsize=(10, 5))

# 绘制运行时间
plt.subplot(1, 2, 1)
plt.plot(k_values, execution_times, marker='o')
plt.xlabel('Number of partitions (K)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Number of Partitions')

# 绘制模型准确性
plt.subplot(1, 2, 2)
for i, accuracies in enumerate(accuracies_list):
    plt.plot([i + 1] * len(accuracies), accuracies, marker='o', linestyle='', label=f'K={i + 2}')
plt.xlabel('Number of partitions (K)')
plt.ylabel('Model Accuracy')
plt.title('Model Accuracy vs Number of Partitions')
plt.legend()

plt.tight_layout()
plt.show()
