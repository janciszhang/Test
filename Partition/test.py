import metis
import networkx as nx

G = nx.karate_club_graph()
(edgecuts, parts) = metis.part_graph(G, nparts=4)
print("Metis edge cuts:", edgecuts)
print("Metis partitions:", parts)

import networkx as nx

G = nx.karate_club_graph()
part = nx.algorithms.community.kernighan_lin_bisection(G)
print("Kernighan-Lin partitions:", part)


import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh

G = nx.karate_club_graph()
L = nx.normalized_laplacian_matrix(G).astype(float)
eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')
second_smallest_eigenvector = eigenvectors[:, 1]
partition = np.where(second_smallest_eigenvector > 0, 0, 1)
print("Spectral partitioning:", partition)


import networkx as nx
from networkx.algorithms.community import girvan_newman

G = nx.karate_club_graph()
comp = girvan_newman(G)
partitions = tuple(sorted(c) for c in next(comp))
print("Girvan-Newman partitions:", partitions)


import community as community_louvain
import networkx as nx

G = nx.karate_club_graph()
partition = community_louvain.best_partition(G)
print("Louvain partitions:", partition)

import time
import networkx as nx
import community as community_louvain
from networkx.algorithms.community import modularity

# 生成图
G = nx.karate_club_graph()

# Metis
start_time = time.time()
(edgecuts, metis_parts) = metis.part_graph(G, nparts=4)
metis_modularity = modularity(G, [{i for i, p in enumerate(metis_parts) if p == part} for part in set(metis_parts)])
metis_time = time.time() - start_time

# Kernighan-Lin
start_time = time.time()
kl_parts = nx.algorithms.community.kernighan_lin_bisection(G)
kl_modularity = modularity(G, kl_parts)
kl_time = time.time() - start_time

# Spectral
start_time = time.time()
L = nx.normalized_laplacian_matrix(G).astype(float)
eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')
second_smallest_eigenvector = eigenvectors[:, 1]
spectral_partition = np.where(second_smallest_eigenvector > 0, 0, 1)
spectral_modularity = modularity(G, [set(np.where(spectral_partition == i)[0]) for i in np.unique(spectral_partition)])
spectral_time = time.time() - start_time

# Girvan-Newman
start_time = time.time()
gn_comp = girvan_newman(G)
gn_parts = tuple(sorted(c) for c in next(gn_comp))
gn_modularity = modularity(G, gn_parts)
gn_time = time.time() - start_time

# Louvain
start_time = time.time()
louvain_partition = community_louvain.best_partition(G)
louvain_parts = {i: [] for i in set(louvain_partition.values())}
for node, part in louvain_partition.items():
    louvain_parts[part].append(node)
louvain_parts = list(louvain_parts.values())
louvain_modularity = modularity(G, louvain_parts)
louvain_time = time.time() - start_time

# 打印结果
print(f"Metis - Modularity: {metis_modularity}, Time: {metis_time}")
print(f"Kernighan-Lin - Modularity: {kl_modularity}, Time: {kl_time}")
print(f"Spectral - Modularity: {spectral_modularity}, Time: {spectral_time}")
print(f"Girvan-Newman - Modularity: {gn_modularity}, Time: {gn_time}")
print(f"Louvain - Modularity: {louvain_modularity}, Time: {louvain_time}")
