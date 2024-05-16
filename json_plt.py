# 生成内存占用图表展示
import json
import matplotlib.pyplot as plt

# 读取导出的内存占用的 Chrome Trace 文件
with open("memory_trace1.json", "r") as f:
    trace_data = json.load(f)

# 提取内存占用数据并生成图表
timestamps = [event["ts"] for event in trace_data["traceEvents"] if event["name"] == "[memory]"]
print(timestamps)
memory_usages = [event["args"]["Bytes"] for event in trace_data["traceEvents"] if event["name"] == "[memory]"]


plt.plot(timestamps, memory_usages)
plt.xlabel("Time (ms)")
plt.ylabel("Memory Usage (bytes)")
plt.title("Memory Usage Over Time")
plt.show()