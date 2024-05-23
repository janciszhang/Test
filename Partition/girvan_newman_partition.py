import time
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community

# 生成一个简单的图数据（这里以 Karate Club 图为例）
G = nx.karate_club_graph()

# 定义不同的分割数量K
k_values = [k for k in range(1,10)]

# 记录不同K值下的模型准确性和运行时间
accuracies = []
runtimes = []

for k in k_values:
    start_time = time.time()
    partitions = next(community.girvan_newman(G))
    end_time = time.time()

    # 根据分割结果计算模型准确性（这里简单地假设每个社区对应一个类别）
    accuracy = len(partitions)  # 模型准确性的一个简单度量（实际情况需根据具体任务来确定）

    accuracies.append(accuracy)
    runtimes.append(end_time - start_time)

# 可视化结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('Number of partitions (K)')
plt.ylabel('Model Accuracy')
plt.title('Model Accuracy vs. Number of partitions')

plt.subplot(1, 2, 2)
plt.plot(k_values, runtimes, marker='o')
plt.xlabel('Number of partitions (K)')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime vs. Number of partitions')

plt.tight_layout()
plt.show()
