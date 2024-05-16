import dgl
import torch
import time
import resource
import sys
import pandas as pd
import matplotlib.pyplot as plt
from thop import profile

n = 0
def memory_usage():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # in bytes


# 定义一个简单的 GNN 模型
class GNNModel(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers):
        super(GNNModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_feats, hidden_feats))
        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(hidden_feats, hidden_feats))
        self.layers.append(torch.nn.Linear(hidden_feats, out_feats))

    def forward(self, g, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


# 定义生成不同大小和复杂度的图的函数
def generate_graphs(num_graphs, num_nodes_per_graph, num_edges_per_graph, num_features):
    graphs = []
    for _ in range(num_graphs):
        g = dgl.DGLGraph()
        g.add_nodes(num_nodes_per_graph)
        src = torch.randint(0, num_nodes_per_graph, (num_edges_per_graph,))
        dst = torch.randint(0, num_nodes_per_graph, (num_edges_per_graph,))
        g.add_edges(src, dst)
        x = torch.randn(num_nodes_per_graph, num_features)
        g.ndata['feat'] = x
        graphs.append(g)
    return graphs



# 定义测试函数
def stress_test(model, graphs, device):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    results = []
    for graph in graphs:
        graph = graph.to(device)
        x = graph.ndata['feat'].to(device)
        start_time = time.time()
        with torch.no_grad():
            _ = model(graph, x)
        end_time = time.time()
        elapsed_time = end_time - start_time

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 模型的总参数数量

        results.append((graph.number_of_nodes(), graph.number_of_edges(), elapsed_time, _.shape, total_params,
                        torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device), memory_usage()))

        # 使用 PyTorch Profiler 进行性能分析
        with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True, record_shapes=True, with_stack=True) as prof:
            # 在这里运行模型或代码
            _ = model(graph, x)

        # 打印 PyTorch Profiler 的性能分析结果
        print(prof)
        # 导出内存占用的 Chrome Trace 文件
        global n
        n = n+1
        prof.export_chrome_trace("memory_trace"+str(n)+".json")


    return results


def small_test(num_graphs, num_nodes_per_graph, num_edges_per_graph, num_features, num_layers):
    # 定义参数
    # num_graphs = 3
    # num_nodes_per_graph = 1000
    # num_edges_per_graph = 50000
    # num_features = 100
    graph_sizes = [(num_nodes_per_graph, num_edges_per_graph)] * num_graphs  # 每个图都采用相同的规模

    # 初始化 GNN 模型
    in_feats = num_features
    hidden_feats = 64
    out_feats = 1
    # num_layers = 3
    model = GNNModel(in_feats, hidden_feats, out_feats, num_layers)

    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 生成图
    graphs = generate_graphs(num_graphs, num_nodes_per_graph, num_edges_per_graph, num_features)

    # 进行压力测试
    results = stress_test(model, graphs, device)

    # 打印结果
    # for result in results:
    #     print(
    #         f"Graph Size: {result[0]} nodes, {result[1]} edges, Elapsed Time: {result[2]} seconds, Output shape: {result[3]}")
    return results


if __name__ == '__main__':
    all_results = []
    test_paras = [(3, 10000, 50000, 10, 3), (3, 10000, 50000, 50, 3), (3, 10000, 50000, 100, 3),
                  (3,10000, 50000, 200, 3),(1, 10000, 50000, 1000, 3),(1, 10000, 50000, 2000, 3)]
    for test_param in test_paras:
        results = small_test(test_param[0], test_param[1], test_param[2], test_param[3], test_param[4])
        all_results.extend(results)
        print(test_param,': done')

    # 将结果放入 DataFrame 中
    df = pd.DataFrame(all_results, columns=['Nodes', 'Edges', 'Elapsed Time', 'shape', 'Total trainable parameters',
                                            'GPU memory allocated', 'GPU memory cached', 'Current memory usage'])
    print(df)
    df.to_csv('stress_test_all_results.csv', index=False)

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(df['Total trainable parameters'], df['Elapsed Time'], marker='o')
    plt.xlabel('Total trainable parameters')
    plt.ylabel('Elapsed Time (seconds)')
    plt.title('Elapsed Time vs Total trainable parameters')
    plt.grid(True)
    plt.show()
