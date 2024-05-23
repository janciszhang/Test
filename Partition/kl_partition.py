import time
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import kernighan_lin_bisection

# 创建一个简单的图
G = nx.karate_club_graph()

# 定义KL分割函数
def kl_partition(graph, k):
    return kernighan_lin_bisection(graph, max_iter=k)

# 测试不同分割数量K的运行速度和模型准确性
K_values = [2, 3, 4, 5]  # 不同的K值
execution_times = []
accuracies = []

for k in K_values:
    start_time = time.time()
    partitions = kl_partition(G, k)
    end_time = time.time()
    execution_times.append(end_time - start_time)
    # 计算模型准确性，这里简单使用分割结果的数量来表示
    accuracies.append(len(partitions))

# 可视化结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(K_values, execution_times, marker='o')
plt.xlabel('Number of Partitions (K)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Number of Partitions')

plt.subplot(1, 2, 2)
plt.plot(K_values, accuracies, marker='o')
plt.xlabel('Number of Partitions (K)')
plt.ylabel('Model Accuracy (Number of Clusters)')
plt.title('Model Accuracy vs Number of Partitions')

plt.tight_layout()
plt.show()
